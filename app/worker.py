import os
import io
import base64
import logging
import tempfile
import requests
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_image(url: str, index: int) -> tuple[int, np.ndarray | None]:
    """Download a single image from URL and return as numpy array."""
    try:
        logger.info(f"[Worker] Downloading image {index + 1}: {url[:80]}...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        img_array = np.frombuffer(response.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            logger.error(f"[Worker] Failed to decode image {index + 1}")
            return (index, None)

        logger.info(f"[Worker] Image {index + 1} downloaded: {img.shape}")
        return (index, img)
    except Exception as e:
        logger.error(f"[Worker] Download failed for image {index + 1}: {e}")
        return (index, None)


def align_images(images: list[np.ndarray]) -> list[np.ndarray]:
    """Align bracketed exposures using MTB alignment."""
    try:
        logger.info("[Worker] Aligning images with MTB...")
        align_mtb = cv2.createAlignMTB()
        aligned = images.copy()
        align_mtb.process(aligned, aligned)
        logger.info("[Worker] Alignment complete")
        return aligned
    except Exception as e:
        logger.warning(f"[Worker] Alignment failed, using originals: {e}")
        return images


def merge_hdr_with_weighting(
    images: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """
    Enhanced Mertens exposure fusion with bracket weighting.
    Gives more weight to the middle (best-exposed) frame.
    """
    n = len(images)

    if weights is None:
        # Default weighting: emphasize middle exposure
        if n == 3:
            weights = [0.8, 1.0, 0.8]
        elif n == 5:
            weights = [0.6, 0.85, 1.0, 0.85, 0.6]
        elif n == 7:
            weights = [0.5, 0.7, 0.85, 1.0, 0.85, 0.7, 0.5]
        else:
            weights = [1.0] * n

    # Ensure weights list matches image count
    if len(weights) != n:
        logger.warning(f"[Worker] Weight count ({len(weights)}) != image count ({n}), using uniform weights")
        weights = [1.0] * n

    logger.info(f"[Worker] Merging {n} images with weights: {weights}")

    # Create Mertens merge with tuned parameters
    merge = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=0.0,  # We handle exposure weighting manually
    )

    # Apply per-frame exposure weights before merge
    weighted_images = []
    for img, w in zip(images, weights):
        float_img = img.astype(np.float32) / 255.0
        weighted_img = np.clip(float_img * w, 0, 1)
        weighted_images.append((weighted_img * 255).astype(np.uint8))

    # Run Mertens fusion
    result = merge.process(weighted_images)

    # Clip and convert to 8-bit
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    # Post-merge: CLAHE for local contrast enhancement
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    result = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    # Gentle unsharp mask for initial clarity
    gaussian = cv2.GaussianBlur(result, (0, 0), 3.0)
    result = cv2.addWeighted(result, 1.3, gaussian, -0.3, 0)

    return result


def send_webhook(
    webhook_url: str,
    webhook_secret: str | None,
    job_id: str,
    status: str,
    output_url: str | None = None,
    result_b64: str | None = None,
    error: str | None = None,
):
    """Send result back to Supabase via webhook."""
    headers = {"Content-Type": "application/json"}
    if webhook_secret:
        headers["x-webhook-secret"] = webhook_secret

    payload: dict = {
        "jobId": job_id,
        "status": status,
    }
    if output_url:
        payload["outputUrl"] = output_url
    if result_b64:
        payload["result"] = result_b64
    if error:
        payload["error"] = error

    try:
        logger.info(f"[Worker] Sending webhook to {webhook_url[:60]}... status={status}")
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=120)
        logger.info(f"[Worker] Webhook response: {resp.status_code}")
        if resp.status_code != 200:
            logger.error(f"[Worker] Webhook body: {resp.text[:300]}")
    except Exception as e:
        logger.error(f"[Worker] Webhook delivery failed: {e}")


def process_job(payload: dict):
    """
    Main job processor. Receives a single dict payload from RQ.
    
    Expected keys:
      - job_id: str
      - input_urls: list[str]
      - webhook_url: str | None
      - webhook_secret: str | None
      - weights: list[float] | None
    """
    job_id = payload["job_id"]
    input_urls = payload["input_urls"]
    webhook_url = payload.get("webhook_url")
    webhook_secret = payload.get("webhook_secret")
    weights = payload.get("weights")

    logger.info(f"[Worker] Processing job {job_id} with {len(input_urls)} images")

    try:
        # 1. Download all images in parallel
        images_indexed: list[tuple[int, np.ndarray | None]] = []
        with ThreadPoolExecutor(max_workers=min(len(input_urls), 5)) as executor:
            futures = {
                executor.submit(download_image, url, i): i
                for i, url in enumerate(input_urls)
            }
            for future in as_completed(futures):
                images_indexed.append(future.result())

        # Sort by original index to maintain bracket order
        images_indexed.sort(key=lambda x: x[0])
        images = [img for _, img in images_indexed if img is not None]

        if len(images) < 2:
            raise ValueError(f"Only {len(images)} images downloaded successfully, need at least 2")

        logger.info(f"[Worker] Downloaded {len(images)}/{len(input_urls)} images successfully")

        # 2. Resize to common dimensions (use the smallest image as reference)
        min_h = min(img.shape[0] for img in images)
        min_w = min(img.shape[1] for img in images)

        # Cap at 4000px on longest side for performance
        max_dim = 4000
        scale = min(1.0, max_dim / max(min_h, min_w))
        target_h = int(min_h * scale)
        target_w = int(min_w * scale)

        resized = [cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA) for img in images]
        logger.info(f"[Worker] Resized to {target_w}x{target_h}")

        # 3. Align exposures
        aligned = align_images(resized)

        # 4. Merge with bracket weighting
        merged = merge_hdr_with_weighting(aligned, weights)
        logger.info(f"[Worker] Merge complete: {merged.shape}")

        # 5. Encode result as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        success, encoded = cv2.imencode(".jpg", merged, encode_params)

        if not success:
            raise ValueError("Failed to encode merged image")

        result_bytes = encoded.tobytes()
        result_b64 = base64.b64encode(result_bytes).decode("utf-8")

        logger.info(f"[Worker] Encoded result: {len(result_bytes) / 1024:.1f} KB")

        # 6. Send webhook with result
        if webhook_url:
            send_webhook(
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
                job_id=job_id,
                status="completed",
                result_b64=result_b64,
            )

        return {"status": "completed", "size_kb": len(result_bytes) / 1024}

    except Exception as e:
        logger.error(f"[Worker] Job {job_id} failed: {e}", exc_info=True)

        if webhook_url:
            send_webhook(
                webhook_url=webhook_url,
                webhook_secret=webhook_secret,
                job_id=job_id,
                status="failed",
                error=str(e),
            )

        raise
