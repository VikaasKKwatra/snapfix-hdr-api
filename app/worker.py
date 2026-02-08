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
    callback_url = job_data.get("callbackUrl")
    
    print(f"Processing job {job_id} with {len(input_urls)} images, style: {style}")
    
    try:
        # TODO: Add your HDR processing logic here
        # 1. Download images from input_urls
        # 2. Merge them using your HDR algorithm  
        # 3. Upload result and get output_url
        
        output_url = "YOUR_RESULT_URL"  # Replace with actual result
        
        # Notify success
        notify_callback(callback_url, job_id, "completed", output_url=output_url)
        return {"status": "completed", "outputUrl": output_url}
        
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        notify_callback(callback_url, job_id, "failed", error=str(e))
        raise

def main():
    redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_PRIVATE_URL") or os.getenv("REDIS_PUBLIC_URL")
    if not redis_url:
        raise RuntimeError("Missing REDIS_URL (or REDIS_PRIVATE_URL / REDIS_PUBLIC_URL)")
    
    conn = redis.from_url(redis_url)
    queue_name = os.getenv("RQ_QUEUE", "default")
    q = Queue(queue_name, connection=conn)
    
    worker = Worker([q], connection=conn)
    worker.work()

if __name__ == "__main__":
    main()
