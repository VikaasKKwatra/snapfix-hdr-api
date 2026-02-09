import os
import base64

import cv2
import numpy as np
import redis
import requests
from rq import Worker, Queue


def notify_callback(callback_url, job_id, status, output_url=None, error=None, webhook_secret=""):
    """Send webhook notification when job completes"""
    if not callback_url:
        print(f"No callback URL for job {job_id}")
        return

    try:
        response = requests.post(callback_url, json={
            "jobId": job_id,
            "status": status,
            "outputUrl": output_url,
            "error": error
        }, headers={
            "Content-Type": "application/json",
            "x-webhook-secret": webhook_secret,
        }, timeout=30)
        print(f"Callback sent for {job_id}: {status} (HTTP {response.status_code})")
    except Exception as e:
        print(f"Callback failed for {job_id}: {e}")


def download_image(url):
    """Download image from URL and return as OpenCV array"""
    url = str(url)
    print(f"Downloading from URL: {url[:100]}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    image_data = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return img


def merge_hdr_mertens(images):
    """Merge bracketed images using Mertens fusion (no exposure times needed)"""
    merge_mertens = cv2.createMergeMertens()
    fusion = merge_mertens.process(images)
    fusion_8bit = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion_8bit


def process_job(job_data):
    """Process HDR merge job"""
    job_id = str(job_data["jobId"])
    input_urls = [str(u) for u in job_data["inputUrls"]]
    style = str(job_data.get("style", "natural"))
    callback_url = job_data.get("webhookUrl", None)
    webhook_secret = job_data.get("webhookSecret", "") or ""

    if callback_url is not None:
        callback_url = str(callback_url)

    print(f"Processing job {job_id} with {len(input_urls)} images, style: {style}")

    try:
        images = []
        for i, url in enumerate(input_urls):
            print(f"Downloading image {i+1}/{len(input_urls)}")
            img = download_image(url)
            if img is not None:
                images.append(img)

        if len(images) < 2:
            raise ValueError(f"Need at least 2 images, got {len(images)}")

        print(f"Merging {len(images)} images...")
        result = merge_hdr_mertens(images)

        if style == "detailed":
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        ok, buffer = cv2.imencode(".jpg", result, encode_params)
        if not ok:
            raise RuntimeError("Failed to encode JPG")

        b64 = base64.b64encode(buffer).decode("utf-8")
        output_url = f"data:image/jpeg;base64,{b64}"

        print(f"Job {job_id} completed successfully")
        notify_callback(callback_url, job_id, "completed", output_url=output_url, webhook_secret=webhook_secret)

        return {"status": "completed", "outputUrl": output_url}

    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        notify_callback(callback_url, job_id, "failed", error=str(e), webhook_secret=webhook_secret)
        raise


def main():
    redis_url = os.getenv("REDIS_URL") or os.getenv("REDIS_PRIVATE_URL") or os.getenv("REDIS_PUBLIC_URL")
    if not redis_url:
        raise RuntimeError("Missing REDIS_URL")

    conn = redis.from_url(redis_url)
    queue_name = os.getenv("RQ_QUEUE", "default")
    q = Queue(queue_name, connection=conn)

    worker = Worker([q], connection=conn)
    worker.work()


if __name__ == "__main__":
    main()
