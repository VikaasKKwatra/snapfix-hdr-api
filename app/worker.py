import os
import cv2
import numpy as np
import requests
import json
import tempfile

def process_job(job_data):
    """HDR merge with maximum quality output."""
    job_id = job_data["jobId"]
    input_urls = job_data["inputUrls"]
    style = job_data.get("style", "natural")
    webhook_url = job_data.get("webhookUrl")
    webhook_secret = job_data.get("webhookSecret")

    print(f"[Worker] Processing job {job_id}, style={style}, {len(input_urls)} images")

    try:
        # 1. Download images at FULL resolution
        images = []
        for i, url in enumerate(input_urls):
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image {i}")
            print(f"[Worker] Image {i}: {img.shape[1]}x{img.shape[0]}")
            images.append(img)

        # 2. Ensure all images are the same size (resize to smallest)
        min_h = min(img.shape[0] for img in images)
        min_w = min(img.shape[1] for img in images)
        aligned = []
        for img in images:
            if img.shape[0] != min_h or img.shape[1] != min_w:
                img = cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_LANCZOS4)
            aligned.append(img)

        # 3. Mertens exposure fusion (works on LDR inputs, no tone mapping needed)
        merge = cv2.createMergeMertens(
            contrast_weight=1.0,
            saturation_weight=1.0,
            exposure_weight=1.0
        )
        fusion = merge.process(aligned)

        # 4. Clip and convert to 8-bit
        fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)

        # 5. Post-processing for best quality
        # Apply CLAHE to luminance channel for better local contrast
        lab = cv2.cvtColor(fusion, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if style == "detailed":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        fusion = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 6. Gentle sharpening via unsharp mask
        gaussian = cv2.GaussianBlur(fusion, (0, 0), sigmaX=2.0)
        fusion = cv2.addWeighted(fusion, 1.3, gaussian, -0.3, 0)

        # 7. Encode as highest quality JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 98]
        success, encoded = cv2.imencode(".jpg", fusion, encode_params)
        if not success:
            raise ValueError("Failed to encode output image")

        print(f"[Worker] Output size: {len(encoded.tobytes()) / 1024:.0f}KB")

        # 8. Save to disk (for polling fallback)
        out_dir = os.getenv("OUTPUT_DIR", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{job_id}.jpg")
        with open(out_path, "wb") as f:
            f.write(encoded.tobytes())

        # 9. Send webhook with base64 result
        if webhook_url:
            import base64
            b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64}"

            headers = {"Content-Type": "application/json"}
            if webhook_secret:
                headers["x-webhook-secret"] = webhook_secret

            payload = {
                "jobId": job_id,
                "status": "completed",
                "outputUrl": data_url,
            }

            print(f"[Worker] Sending webhook to {webhook_url}")
            wr = requests.post(webhook_url, json=payload, headers=headers, timeout=60)
            print(f"[Worker] Webhook response: {wr.status_code}")

        print(f"[Worker] Job {job_id} completed successfully")

    except Exception as e:
        print(f"[Worker] Job {job_id} FAILED: {e}")
        if webhook_url:
            try:
                headers = {"Content-Type": "application/json"}
                if webhook_secret:
                    headers["x-webhook-secret"] = webhook_secret
                requests.post(webhook_url, json={
                    "jobId": job_id,
                    "status": "failed",
                    "error": str(e),
                }, headers=headers, timeout=30)
            except Exception as we:
                print(f"[Worker] Failed to send error webhook: {we}")
        raise
