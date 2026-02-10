import os
import cv2
import numpy as np
import requests
import logging
import base64
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_image(url: str) -> np.ndarray:
    """Download image from URL and return as OpenCV BGR array."""
    logger.info(f"[Worker] Downloading: {url[:80]}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from {url[:80]}")
    logger.info(f"[Worker] Downloaded image: {img.shape}")
    return img


def align_images(images: list[np.ndarray]) -> list[np.ndarray]:
    """Align bracket images using ECC with ORB fallback."""
    if len(images) < 2:
        return images

    reference = images[len(images) // 2]
    aligned = []

    for i, img in enumerate(images):
        if i == len(images) // 2:
            aligned.append(img)
            continue

        try:
            # ECC alignment
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, img_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria
            )
            aligned_img = cv2.warpAffine(
                img, warp_matrix, (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            logger.info(f"[Worker] Image {i} aligned via ECC")
            aligned.append(aligned_img)

        except cv2.error:
            try:
                # ORB fallback
                orb = cv2.ORB_create(5000)
                kp1, des1 = orb.detectAndCompute(ref_gray, None)
                kp2, des2 = orb.detectAndCompute(img_gray, None)

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)[:50]

                if len(matches) >= 4:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
                    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
                    aligned_img = cv2.warpPerspective(
                        img, H, (reference.shape[1], reference.shape[0])
                    )
                    logger.info(f"[Worker] Image {i} aligned via ORB fallback")
                    aligned.append(aligned_img)
                else:
                    logger.warning(f"[Worker] Image {i} not enough matches, using unaligned")
                    aligned.append(img)
            except Exception as e:
                logger.warning(f"[Worker] Image {i} alignment failed entirely: {e}")
                aligned.append(img)

    return aligned


def merge_hdr_with_weighting(images: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Merge bracket images using Mertens fusion with optional exposure weighting."""
    if weights is None:
        # Default: emphasize middle exposure
        n = len(images)
        if n == 3:
            weights = [0.8, 1.0, 0.8]
        elif n == 5:
            weights = [0.6, 0.8, 1.0, 0.8, 0.6]
        else:
            weights = [1.0] * n

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    # Apply weights to images
    weighted_images = []
    for img, w in zip(images, weights):
        weighted = cv2.convertScaleAbs(img, alpha=w)
        weighted_images.append(weighted)

    # Mertens exposure fusion
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0,
    )
    fusion = merge_mertens.process(weighted_images)

    # Convert from [0,1] float to [0,255] uint8
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)

    return fusion


def apply_clahe(img: np.ndarray, clip_limit: float = 2.5, grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE for local contrast enhancement in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def apply_color_correction(img: np.ndarray) -> np.ndarray:
    """Subtle LAB-space color correction for neutral whites."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Push a and b channels slightly toward neutral
    a = cv2.addWeighted(a, 0.95, np.full_like(a, 128), 0.05, 0)
    b = cv2.addWeighted(b, 0.95, np.full_like(b, 128), 0.05, 0)

    corrected = cv2.merge([l, a, b])
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def apply_sharpening(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Edge-aware sharpening using unsharp mask."""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def apply_denoising(img: np.ndarray) -> np.ndarray:
    """Light denoising - positional args only for OpenCV compatibility."""
    return cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 7, 21)


def process_job(payload: dict):
    """Main HDR processing pipeline."""
    # Support both camelCase and snake_case keys
    job_id = payload.get("jobId") or payload.get("job_id", "")
    input_urls = payload.get("inputUrls") or payload.get("input_urls", [])
    webhook_url = payload.get("webhookUrl") or payload.get("webhook_url")
    webhook_secret = payload.get("webhookSecret") or payload.get("webhook_secret")
    weights = payload.get("weights")
    style = payload.get("style", "natural")

    logger.info(f"[Worker] Processing job {job_id} with {len(input_urls)} images, style={style}")

    try:
        # 1. Download all bracket images in parallel-ish
        images = [download_image(url) for url in input_urls]
        logger.info(f"[Worker] Downloaded {len(images)} images")

        # 2. Align images
        aligned = align_images(images)
        logger.info(f"[Worker] Alignment complete")

        # 3. Weighted Mertens fusion
        merged = merge_hdr_with_weighting(aligned, weights)
        logger.info(f"[Worker] Mertens fusion complete: {merged.shape}")

        # 4. CLAHE local contrast
        enhanced = apply_clahe(merged, clip_limit=2.5, grid_size=8)
        logger.info(f"[Worker] CLAHE applied")

        # 5. Color correction
        enhanced = apply_color_correction(enhanced)
        logger.info(f"[Worker] Color correction applied")

        # 6. Light denoising
        enhanced = apply_denoising(enhanced)
        logger.info(f"[Worker] Denoising applied")

        # 7. Edge-aware sharpening
        enhanced = apply_sharpening(enhanced, strength=0.5)
        logger.info(f"[Worker] Sharpening applied")

        # 8. Encode to JPEG at 97% quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 97]
        success, buffer = cv2.imencode(".jpg", enhanced, encode_params)
        if not success:
            raise ValueError("Failed to encode result image")

        result_bytes = buffer.tobytes()
        logger.info(f"[Worker] Encoded result: {len(result_bytes)} bytes ({len(result_bytes)/1024:.1f}KB)")

        # 9. Save to outputs directory for polling fallback
        out_dir = os.getenv("OUTPUT_DIR", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{job_id}.jpg")
        with open(out_path, "wb") as f:
            f.write(result_bytes)
        logger.info(f"[Worker] Saved to {out_path}")

        # 10. Send result via webhook
        if webhook_url:
            result_b64 = base64.b64encode(result_bytes).decode("utf-8")
            webhook_payload = {
                "jobId": job_id,
                "status": "completed",
                "result": f"data:image/jpeg;base64,{result_b64}",
            }
            headers = {"Content-Type": "application/json"}
            if webhook_secret:
                headers["x-webhook-secret"] = webhook_secret

            logger.info(f"[Worker] Sending webhook to {webhook_url[:60]}... payload size: {len(result_b64)//1024}KB")

            try:
                resp = requests.post(webhook_url, json=webhook_payload, headers=headers, timeout=120)
                logger.info(f"[Worker] Webhook response: {resp.status_code} - {resp.text[:200]}")
            except Exception as e:
                logger.error(f"[Worker] Webhook delivery failed: {e}")
        else:
            logger.info(f"[Worker] No webhook URL, result available via polling only")

        logger.info(f"[Worker] Job {job_id} completed successfully")
        return {"jobId": job_id, "status": "completed"}

    except Exception as e:
        logger.error(f"[Worker] Job {job_id} failed: {e}", exc_info=True)

        # Try to notify webhook of failure
        if webhook_url:
            try:
                fail_payload = {
                    "jobId": job_id,
                    "status": "failed",
                    "error": str(e),
                }
                headers = {"Content-Type": "application/json"}
                if webhook_secret:
                    headers["x-webhook-secret"] = webhook_secret
                requests.post(webhook_url, json=fail_payload, headers=headers, timeout=30)
            except Exception:
                pass

        raise
