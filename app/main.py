import os
import uuid
import redis
from rq import Queue
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

app = FastAPI(title="SnapFix HDR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_conn = redis.from_url(REDIS_URL)

# IMPORTANT: Must match the queue your worker listens to
task_queue = Queue("hdr_jobs", connection=redis_conn)

# Optional auth (set in Railway Variables). If blank, auth is disabled.
SNAPFIX_API_KEY = os.environ.get("SNAPFIX_API_KEY", "")


class HDRJobRequest(BaseModel):
    # job id aliases
    jobId: Optional[str] = Field(None, alias="job_id")
    job_id_alt: Optional[str] = Field(None, alias="jobId")

    # input urls aliases
    inputUrls: Optional[List[str]] = Field(None, alias="input_urls")
    input_urls_alt: Optional[List[str]] = Field(None, alias="inputUrls")

    # style options
    # natural = clean MLS
    # window_soft = safe window pull
    # window_strong = aggressive window pull (halo guarded)
    style: Literal["natural", "window_soft", "window_strong"] = "natural"

    resolution: str = "standard"

    # exposure order option
    order: Literal["random", "dark", "normal", "bright"] = "random"

    # webhook aliases
    webhookUrl: Optional[str] = Field(None, alias="webhook_url")
    webhook_url_alt: Optional[str] = Field(None, alias="webhookUrl")

    webhookSecret: Optional[str] = Field(None, alias="webhook_secret")
    webhook_secret_alt: Optional[str] = Field(None, alias="webhookSecret")

    class Config:
        populate_by_name = True

    @property
    def canonical_job_id(self) -> str:
        return self.jobId or self.job_id_alt or str(uuid.uuid4())

    @property
    def canonical_input_urls(self) -> List[str]:
        return self.inputUrls or self.input_urls_alt or []

    @property
    def canonical_webhook_url(self) -> Optional[str]:
        return self.webhookUrl or self.webhook_url_alt

    @property
    def canonical_webhook_secret(self) -> Optional[str]:
        return self.webhookSecret or self.webhook_secret_alt


def require_auth(request: Request):
    if not SNAPFIX_API_KEY:
        return
    api_key_q = request.query_params.get("api_key")
    api_key_h = request.headers.get("x-api-key")
    if api_key_q != SNAPFIX_API_KEY and api_key_h != SNAPFIX_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/")
def health():
    return {"status": "ok", "service": "snapfix-hdr-api"}


@app.post("/v1/hdr/jobs")
async def create_job(request: Request):
    require_auth(request)

    raw = await request.json()
    job_data = HDRJobRequest(**raw)

    job_id = job_data.canonical_job_id
    input_urls = job_data.canonical_input_urls

    if len(input_urls) < 2:
        raise HTTPException(status_code=400, detail="At least 2 input URLs required")

    payload = {
        "job_id": job_id,
        "input_urls": input_urls,
        "style": job_data.style,
        "resolution": job_data.resolution,
        "order": job_data.order,
        "webhook_url": job_data.canonical_webhook_url,
        "webhook_secret": job_data.canonical_webhook_secret,
    }

    task_queue.enqueue(
        "app.worker.process_job",
        payload,
        job_id=f"hdr-{job_id}",
        job_timeout=900,
        ttl=3600,
        result_ttl=3600,
    )

    print(f"[API] Enqueued job {job_id} ({len(input_urls)} imgs) style={job_data.style} order={job_data.order}")

    return {
        "jobId": job_id,
        "job_id": job_id,
        "status": "queued",
        "message": "Job enqueued for processing",
    }


@app.get("/v1/hdr/result/{job_id}")
async def get_result(job_id: str, request: Request):
    require_auth(request)

    job_key = f"hdr_result:{job_id}"
    result = redis_conn.hgetall(job_key)

    if not result:
        raise HTTPException(status_code=404, detail="Not ready or job not found")

    status = result.get(b"status", b"unknown").decode()

    if status == "completed":
        output_url = result.get(b"output_url", b"").decode()
        return {"status": "completed", "outputUrl": output_url, "jobId": job_id}

    if status == "failed":
        error = result.get(b"error", b"Unknown error").decode()
        return {"status": "failed", "error": error, "jobId": job_id}

    return {"status": status, "jobId": job_id}
import io
import rawpy
import numpy as np
import cv2
import requests as http_requests


@app.post("/v1/convert-raw")
async def convert_raw(request: Request):
    require_auth(request)

    body = await request.json()
    raw_url = body.get("rawUrl") or body.get("raw_url")
    quality = int(body.get("quality", 97))

    if not raw_url:
        raise HTTPException(status_code=400, detail="Missing rawUrl")

    print(f"[Convert] Downloading RAW from URL ({len(raw_url)} chars)...")

    resp = http_requests.get(raw_url, timeout=300)
    resp.raise_for_status()
    raw_bytes = resp.content
    print(f"[Convert] Downloaded {len(raw_bytes) / 1024 / 1024:.1f}MB RAW file")

    try:
        raw = rawpy.imread(io.BytesIO(raw_bytes))
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=False,
            output_bps=8,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
        raw.close()
    except Exception as e:
        print(f"[Convert] rawpy failed: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to decode RAW file: {str(e)}")

    print(f"[Convert] Rendered: {rgb.shape[1]}x{rgb.shape[0]} pixels")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, jpeg_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encoding failed")

    jpeg_bytes = jpeg_buf.tobytes()
    print(f"[Convert] Output JPEG: {len(jpeg_bytes) / 1024 / 1024:.1f}MB at q={quality}")

    from fastapi.responses import Response
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Content-Length": str(len(jpeg_bytes)),
            "X-Image-Width": str(rgb.shape[1]),
            "X-Image-Height": str(rgb.shape[0]),
        },
    )
