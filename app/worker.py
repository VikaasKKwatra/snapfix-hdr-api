import cv2
import numpy as np
import os
import requests
import base64
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hdr-worker")


def download_image(url: str) -> np.ndarray:
    """Download image from signed URL and return as numpy array."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from URL")
    return img


def align_images(images: list[np.ndarray]) -> list[np.ndarray]:
    """Align all images to the first image using ECC algorithm for sub-pixel accuracy."""
    if len(images) < 2:
        return images

    reference = images[0]
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    h, w = reference.shape[:2]
    aligned = [reference]

    # ECC alignment parameters
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    for i, img in enumerate(images[1:], 1):
        try:
            # Resize to match reference
            img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # Try ECC alignment (sub-pixel accurate)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                _, warp_matrix = cv2.findTransformECC(
                    ref_gray, img_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
                )
            except cv2.error:
                # Fallback to ORB feature matching
                logger.warning(f"ECC failed for image {i}, trying ORB fallback")
                orb = cv2.ORB_create(5000)
                kp1, des1 = orb.detectAndCompute(ref_gray, None)
                kp2, des2 = orb.detectAndCompute(img_gray, None)
                if des1 is not None and des2 is not None:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)[:50]
                    if len(matches) >= 4:
                        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                        if M is not None:
                            warp_matrix = M.astype(np.float32)

            aligned_img = cv2.warpAffine(
                img_resized, warp_matrix, (w, h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT_101
            )
            aligned.append(aligned_img)
            logger.info(f"Aligned image {i} successfully")
        except Exception as e:
            logger.warning(f"Alignment failed for image {i}: {e}, using resized version")
            aligned.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4))

    return aligned


def exposure_fusion(images: list[np.ndarray]) -> np.ndarray:
    """Merge multiple exposures using Mertens fusion."""
    merge = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0
    )
    fusion = merge.process(images)
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion


def window_pull(images: list[np.ndarray], fused: np.ndarray) -> np.ndarray:
    """Recover blown highlights (windows) by blending darker exposures."""
    # Find the darkest exposure (best for window detail)
    darkest = min(images, key=lambda img: np.mean(img))

    # Create luminosity mask for bright areas
    fused_gray = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(fused_gray, 220, 255, cv2.THRESH_BINARY)

    # Feather the mask for smooth blending
    bright_mask = cv2.GaussianBlur(bright_mask, (51, 51), 0)
    mask_float = bright_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_float] * 3, axis=-1)

    # Resize darkest to match fused
    h, w = fused.shape[:2]
    darkest_resized = cv2.resize(darkest, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Blend: use darkest exposure in bright areas
    result = (fused.astype(np.float32) * (1 - mask_3ch * 0.7) +
              darkest_resized.astype(np.float32) * mask_3ch * 0.7)

    return np.clip(result, 0, 255).astype(np.uint8)


def color_correction(img: np.ndarray) -> np.ndarray:
    """Gray-world color correction in LAB space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Only correct if there's a noticeable cast
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])

    # 128 is neutral in LAB a/b channels
    a_shift = 128 - avg_a
    b_shift = 128 - avg_b

    # Apply gentle correction (don't fully neutralize - keep some warmth)
    correction_strength = 0.5
    lab[:, :, 1] += a_shift * correction_strength
    lab[:, :, 2] += b_shift * correction_strength

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def smart_sharpen(img: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Unsharp mask with edge-aware sharpening."""
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise_light(img: np.ndarray) -> np.ndarray:
    """Light denoising that preserves detail."""
    return cv2.fastNlMeansDenoisingColored(img, None, h=3, hForColorComponents=3,
                                           templateWindowSize=7, searchWindowSize=21)


def process_job(data: dict):
    """
    Full HDR processing pipeline:
    1. Download & align images
    2. Exposure fusion (Mertens)
    3. Window pull (recover blown highlights)
    4. Color correction (neutral whites)
    5. Local contrast (CLAHE)
    6. Light denoise + smart sharpen
    7. High-quality JPEG output
    """
    job_id = data.get("job_id", "")
    image_urls = data.get("image_urls", [])
    style = data.get("style", "natural")
    webhook_url = data.get("webhook_url")
    webhook_secret = data.get("webhook_secret")

    logger.info(f"[{job_id}] Starting HDR processing, {len(image_urls)} images, style={style}")

    try:
        # 1. Download images
        images = []
        for i, url in enumerate(image_urls):
            img = download_image(url)
            logger.info(f"[{job_id}] Downloaded image {i+1}: {img.shape}")
            images.append(img)

        if len(images) < 2:
            raise ValueError("Need at least 2 images for HDR merge")

        # 2. Align
        logger.info(f"[{job_id}] Aligning images...")
        aligned = align_images(images)

        # 3. Exposure fusion
        logger.info(f"[{job_id}] Running exposure fusion...")
        fused = exposure_fusion(aligned)

        # 4. Window pull
        logger.info(f"[{job_id}] Applying window pull...")
        pulled = window_pull(aligned, fused)

        # 5. Color correction
        logger.info(f"[{job_id}] Color correction...")
        corrected = color_correction(pulled)

        # 6. CLAHE local contrast
        logger.info(f"[{job_id}] Applying CLAHE...")
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 7. Denoise + sharpen
        logger.info(f"[{job_id}] Denoising and sharpening...")
        denoised = denoise_light(enhanced)
        final = smart_sharpen(denoised, strength=0.4)

        # Encode as high-quality JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        _, buffer = cv2.imencode('.jpg', final, encode_params)
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        logger.info(f"[{job_id}] HDR processing complete, output size: {len(result_b64)} chars")

        # Also save to outputs/ for direct download
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", f"{job_id}.jpg")
        with open(output_path, "wb") as f:
            f.write(buffer)
        logger.info(f"[{job_id}] Saved to {output_path}")

        # Send webhook
        if webhook_url:
            logger.info(f"[{job_id}] Sending webhook to {webhook_url}")
            headers = {"Content-Type": "application/json"}
            if webhook_secret:
                headers["x-webhook-secret"] = webhook_secret
            
            payload = {
                "jobId": job_id,
                "status": "completed",
                "result": result_b64
            }
            
            try:
                resp = requests.post(webhook_url, json=payload, headers=headers, timeout=120)
                logger.info(f"[{job_id}] Webhook response: {resp.status_code}")
                if resp.status_code != 200:
                    logger.error(f"[{job_id}] Webhook body: {resp.text[:500]}")
            except Exception as e:
                logger.error(f"[{job_id}] Webhook failed: {e}")

        return {"status": "completed", "job_id": job_id}

    except Exception as e:
        logger.error(f"[{job_id}] Processing failed: {e}")

        # Send failure webhook
        if webhook_url:
            try:
                headers = {"Content-Type": "application/json"}
                if webhook_secret:
                    headers["x-webhook-secret"] = webhook_secret
                requests.post(webhook_url, json={
                    "jobId": job_id,
                    "status": "failed",
                    "error": str(e)
                }, headers=headers, timeout=30)
            except Exception:
                pass

        raise
