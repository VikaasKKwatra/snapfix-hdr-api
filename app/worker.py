import os
import redis
import requests
from rq import Worker, Queue

def notify_callback(callback_url, job_id, status, output_url=None, error=None):
    """Send webhook notification when job completes"""
    if not callback_url:
        print(f"No callback URL for job {job_id}, skipping notification")
        return
    
    try:
        response = requests.post(callback_url, json={
            "jobId": job_id,
            "status": status,
            "outputUrl": output_url,
            "error": error
        }, headers={"Content-Type": "application/json"}, timeout=30)
        print(f"Callback sent for {job_id}: {status} (HTTP {response.status_code})")
    except Exception as e:
        print(f"Callback failed for {job_id}: {e}")

def process_job(job_data):
    """Process HDR merge job - called by RQ worker"""
    job_id = job_data["jobId"]
    input_urls = job_data["inputUrls"]
    style = job_data.get("style", "natural")
