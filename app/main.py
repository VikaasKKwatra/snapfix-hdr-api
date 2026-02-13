import os
import uuid
import redis
import random
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
task_queue = Queue("hdr_jobs", connection=redis_conn)


class HDRJobRequest(BaseModel):
    jobId: Optional[str] = Field(None, alias="job_id")
    job_id_alt: Optional[str] = Field(None, alias="jobId")

    inputUrls: Optional[List[str]] = Field(None, alias="input_urls")
    input_urls_alt: Optional[List[str]] = Field(None, alias="inputUrls")

    style: str = "natural"
    resolution: str = "standard"

    # NEW: order of exposures
    order: Literal["as_provided", "dark-normal-bright", "random"] = "as_provided"

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


@app.get("/")
def health():
    return {"status": "ok", "service": "snapfix-hdr-api"}


@app.post("/v1/hdr/jobs")
async def create_job(request: Request):
    raw = await request.json()
    job_data = HDRJobRequest(**raw)

    job_id = job_data.canonical_job_id
    input_urls = job_data.canonical_input_urls

    if len(input_urls) < 2:
        raise HTTPException(status_code=400, detail="At least 2 input URLs required")

    # NEW: order handling
    order = job_data.order
    urls = list(input_urls)

    if order == "random":
        random.shuffle(urls)

    # NOTE: "dark-normal-bright" assumes caller already passed in that order.
    # If you want auto-detect darkest/brightest from images, we do it in worker after download.

    payload = {
        "job_id": job_id,
        "input_urls": urls,
        "style": job_data.style,
        "resolution": job_data.resolution,
        "order": order,
        "webhook_url": job_data.canonical_webhook_url,
        "webhook_secret": job_data.canonical_webhook_secret,
    }

    task_queue.enqueue(
        "app.worker.process_job",
        payload,
        job_id=f"hdr-{job_id}",
        job_timeout=900,
    )

    print(f"[API] Enqueued job {job_id} with {len(urls)} inputs (order={order})")

    return {
        "jobId": job_id,
        "job_id": job_id,
        "status": "queued",
        "message": "Job enqueued for processing",
    }


@app.get("/v1/hdr/result/{job_id}")
async def get_result(job_id: str):
    job_key = f"hdr_result:{job_id}"
    result = redis_conn.hgetall(job_key)

    if not result:
        raise HTTPException(status_code=404, detail="Not ready or job not found")

    status = result.get(b"status", b"unknown").decode()

    if status == "completed":
        output_url = result.get(b"output_url", b"").decode()
        return {"status": "completed", "outputUrl": output_url, "jobId": job_id}
    elif status == "failed":
        error = result.get(b"error", b"Unknown error").decode()
        return {"status": "failed", "error": error, "jobId": job_id}
    else:
        return {"status": status, "jobId": job_id}
