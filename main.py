import os
import uuid
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HDR Merge Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis / RQ setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_conn = Redis.from_url(REDIS_URL)
task_queue = Queue("hdr_jobs", connection=redis_conn)


class HDRJobRequest(BaseModel):
    job_id: Optional[str] = Field(None, alias="jobId")
    input_urls: list[str] = Field(..., alias="inputUrls")
    webhook_url: Optional[str] = Field(None, alias="webhookUrl")
    webhook_secret: Optional[str] = Field(None, alias="webhookSecret")
    weights: Optional[list[float]] = Field(None, alias="weights")

    class Config:
        populate_by_name = True


class HDRJobResponse(BaseModel):
    job_id: str = Field(..., alias="jobId")
    status: str


@app.get("/health")
async def health():
    return {"status": "ok", "queue_size": len(task_queue)}


@app.post("/merge", response_model=HDRJobResponse)
async def create_merge_job(request: HDRJobRequest):
    job_id = request.job_id or str(uuid.uuid4())

    logger.info(f"[API] New merge job {job_id} with {len(request.input_urls)} images")

    if len(request.input_urls) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 input images")

    if len(request.input_urls) > 9:
        raise HTTPException(status_code=400, detail="Maximum 9 input images")

    # Single dict payload for reliable RQ serialization
    payload = {
        "job_id": job_id,
        "input_urls": request.input_urls,
        "webhook_url": request.webhook_url,
        "webhook_secret": request.webhook_secret,
        "weights": request.weights,
    }

    task_queue.enqueue(
        "app.worker.process_job",
        payload,
        job_id=job_id,
        job_timeout="600s",
        result_ttl=3600,
    )

    logger.info(f"[API] Job {job_id} enqueued successfully")
    return HDRJobResponse(jobId=job_id, status="queued")


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    from rq.job import Job as RQJob
    try:
        rq_job = RQJob.fetch(job_id, connection=redis_conn)
        status_map = {
            "queued": "queued",
            "started": "processing",
            "finished": "completed",
            "failed": "failed",
            "deferred": "queued",
            "scheduled": "queued",
        }
        status = status_map.get(rq_job.get_status(), "unknown")

        result = None
        error = None
        if status == "completed" and rq_job.result:
            result = rq_job.result
        if status == "failed":
            error = str(rq_job.exc_info) if rq_job.exc_info else "Unknown error"

        return {
            "jobId": job_id,
            "status": status,
            "result": result,
            "error": error,
        }
    except Exception as e:
        logger.error(f"[API] Status check failed for {job_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
