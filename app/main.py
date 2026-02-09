import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
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
    # Support both camelCase and snake_case
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

    def get_job_id(self) -> str:
        return self.jobId or self.job_id or ""
    
    def get_input_urls(self) -> List[str]:
        return self.inputUrls or self.input_urls or []
    
    def get_webhook_url(self) -> Optional[str]:
        return self.webhookUrl or self.webhook_url
    
    def get_webhook_secret(self) -> Optional[str]:
        return self.webhookSecret or self.webhook_secret

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/hdr/jobs")
def create_job(req: JobRequest, x_api_key: str = ""):
    require_api_key(x_api_key)

    if not REDIS_URL:
        raise HTTPException(status_code=500, detail="REDIS_URL missing on server.")

    r = redis.from_url(REDIS_URL)
    q = Queue(QUEUE_NAME, connection=r)
    
    # Build normalized job data
    job_data = {
        "jobId": req.get_job_id(),
        "inputUrls": req.get_input_urls(),
        "style": req.style,
        "webhookUrl": req.get_webhook_url(),
        "webhookSecret": req.get_webhook_secret(),
    }
    
    q.enqueue("app.worker.process_job", job_data)

    return {"accepted": True, "jobId": req.get_job_id(), "status": "QUEUED"}

@app.get("/v1/hdr/result/{job_id}")
def get_result(job_id: str):
    out_dir = os.getenv("OUTPUT_DIR", "outputs")
    path = os.path.join(out_dir, f"{job_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not ready")
    return FileResponse(path, media_type="image/jpeg", filename=f"{job_id}.jpg")
