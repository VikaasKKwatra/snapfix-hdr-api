import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from rq import Queue
import redis

app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL") or os.getenv("REDIS_PRIVATE_URL") or os.getenv("REDIS_PUBLIC_URL") or ""
QUEUE_NAME = os.getenv("RQ_QUEUE", "default")
API_KEY = os.getenv("SNAPFIX_API_KEY", "")


class JobRequest(BaseModel):
    jobId: Optional[str] = None
    job_id: Optional[str] = None
    inputUrls: Optional[List[str]] = None
    input_urls: Optional[List[str]] = None
    style: str = "natural"
    resolution: str = "standard"
    webhookUrl: Optional[str] = None
    webhook_url: Optional[str] = None
    webhookSecret: Optional[str] = None
    webhook_secret: Optional[str] = None
    callbackUrl: Optional[str] = None

    @property
    def canonical_job_id(self) -> str:
        return self.jobId or self.job_id or ""

    @property
    def canonical_input_urls(self) -> List[str]:
        return self.inputUrls or self.input_urls or []

    @property
    def canonical_webhook_url(self) -> Optional[str]:
        return self.webhookUrl or self.webhook_url or self.callbackUrl

    @property
    def canonical_webhook_secret(self) -> Optional[str]:
        return self.webhookSecret or self.webhook_secret


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/hdr/jobs")
def create_job(req: JobRequest, x_api_key: str = Query("")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not REDIS_URL:
        raise HTTPException(status_code=500, detail="REDIS_URL missing on server.")

    jid = req.canonical_job_id
    urls = req.canonical_input_urls
    if not jid or len(urls) < 2:
        raise HTTPException(status_code=400, detail="Need jobId and at least 2 inputUrls")

    job_data = {
        "jobId": jid,
        "inputUrls": urls,
        "style": req.style,
        "webhookUrl": req.canonical_webhook_url,
        "webhookSecret": req.canonical_webhook_secret,
    }

    r = redis.from_url(REDIS_URL)
    q = Queue(QUEUE_NAME, connection=r)

    from worker import process_job
    q.enqueue(process_job, job_data, job_id=jid)

    return {"accepted": True, "jobId": jid, "status": "QUEUED"}


@app.get("/v1/hdr/result/{job_id}")
def get_result(job_id: str):
    out_dir = os.getenv("OUTPUT_DIR", "outputs")
    path = os.path.join(out_dir, f"{job_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not ready")
    return FileResponse(path, media_type="image/jpeg", filename=f"{job_id}.jpg")
