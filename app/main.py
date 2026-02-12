import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from redis import Redis
from rq import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("snapfix-hdr-api")

app = FastAPI(title="Snapfix HDR API", version="5.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_conn = Redis.from_url(REDIS_URL)
task_queue = Queue("hdr_jobs", connection=redis_conn)

# In-memory result store (Redis-backed for production)
RESULTS_PREFIX = "hdr_result:"
STATUS_PREFIX = "hdr_status:"


class HDRJobRequest(BaseModel):
    jobId: Optional[str] = Field(None, alias="job_id")
    job_id_alt: Optional[str] = Field(None, alias="jobId")
    inputUrls: Optional[List[str]] = Field(None, alias="input_urls")
    input_urls_alt: Optional[List[str]] = Field(None, alias="inputUrls")
    style: str = "natural"
    resolution: str = "standard"
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


def verify_api_key(x_api_key: Optional[str] = None, authorization: Optional[str] = None):
    expected = os.environ.get("SNAPFIX_API_KEY", "")
    if not expected:
        return  # No key configured, skip auth
    provided = x_api_key or ""
    if not provided and authorization:
        provided = authorization.replace("Bearer ", "")
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "5.2"}


@app.post("/v1/hdr/jobs")
async def create_hdr_job(
    request: HDRJobRequest,
    x_api_key: Optional[str] = Query(None),
    authorization: Optional[str] = None,
):
    verify_api_key(x_api_key, authorization)

    job_id = request.canonical_job_id
    input_urls = request.canonical_input_urls

    if not input_urls or len(input_urls) < 3:
        raise HTTPException(status_code=400, detail="At least 3 input URLs required")

    logger.info(f"[API] Creating job {job_id} with {len(input_urls)} inputs, style={request.style}")

    # Store initial status
    redis_conn.set(f"{STATUS_PREFIX}{job_id}", "processing", ex=3600)

    # Build task payload as single dict
    task_payload = {
        "job_id": job_id,
        "input_urls": input_urls,
        "style": request.style,
        "resolution": request.resolution,
        "webhook_url": request.canonical_webhook_url,
        "webhook_secret": request.canonical_webhook_secret,
    }

    # Enqueue with string import path
    task_queue.enqueue(
        "app.worker.process_job",
        task_payload,
        job_id=f"hdr-{job_id}",
        job_timeout=600,
        result_ttl=3600,
    )

    logger.info(f"[API] Job {job_id} enqueued successfully")

    return {
        "jobId": job_id,
        "job_id": job_id,
        "status": "processing",
        "message": "Job enqueued for processing",
    }


@app.get("/v1/hdr/result/{job_id}")
async def get_hdr_result(
    job_id: str,
    x_api_key: Optional[str] = Query(None),
    authorization: Optional[str] = None,
):
    verify_api_key(x_api_key, authorization)

    status = redis_conn.get(f"{STATUS_PREFIX}{job_id}")
    if status:
        status = status.decode("utf-8")

    result_url = redis_conn.get(f"{RESULTS_PREFIX}{job_id}")
    if result_url:
        result_url = result_url.decode("utf-8")

    if status == "completed" and result_url:
        return {
            "status": "completed",
            "jobId": job_id,
            "outputUrl": result_url,
            "resultUrl": result_url,
        }

    if status == "failed":
        error = redis_conn.get(f"{RESULTS_PREFIX}{job_id}:error")
        error_msg = error.decode("utf-8") if error else "Unknown error"
        return {"status": "failed", "jobId": job_id, "error": error_msg}

    if status == "processing":
        raise HTTPException(status_code=404, detail="Not ready - still processing")

    raise HTTPException(status_code=404, detail="Job not found or not ready")
