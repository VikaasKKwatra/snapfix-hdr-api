import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from rq import Queue
import redis

app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL", "")
QUEUE_NAME = os.getenv("QUEUE_NAME", "hdr")
API_KEY = os.getenv("API_KEY", "")

def require_api_key(x_api_key: str = ""):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

class JobRequest(BaseModel):
    jobId: str
    inputUrls: List[HttpUrl]
    style: str = "natural"
    resolution: str = "standard"
    callbackUrl: Optional[HttpUrl] = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/hdr/jobs")@app.post("/v1/hdr/jobs")
def create_job(req: JobRequest, x_api_key: str = ""):
    require_api_key(x_api_key)

    if not REDIS_URL:
        raise HTTPException(status_code=500, detail="REDIS_URL missing on server.")

    r = redis.from_url(REDIS_URL)
    q = Queue(QUEUE_NAME, connection=r)
    q.enqueue("app.worker.process_job", req.model_dump())

    return {"accepted": True, "jobId": req.jobId, "status": "QUEUED"}

@app.get("/v1/hdr/result/{job_id}")
def get_result(job_id: str):
    out_dir = os.getenv("OUTPUT_DIR", "outputs")
    path = os.path.join(out_dir, f"{job_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not ready")
    return FileResponse(path, media_type="image/jpeg", filename=f"{job_id}.jpg")
