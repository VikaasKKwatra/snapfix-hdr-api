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
        raise ValueError(f"Failed to decode image from URL")
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
                img_aligned = cv2.warpAffine(
                    img_resized, warp_matrix, (w, h),
                    flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT
                )
            except cv2.error:
                # Fallback to ORB feature matching
                logger.warning(f"ECC failed for image {i}, trying ORB feature matching")
                img_aligned = align_with_features(ref_gray, img_gray, img_resized, w, h)

            aligned.append(img_aligned)
            logger.info(f"Aligned image {i}/{len(images)-1}")
        except Exception as e:
            logger.warning(f"Alignment failed for image {i}, using resized: {e}")
            aligned.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4))

    return aligned


def align_with_features(ref_gray, img_gray, img_color, w, h):
    """Feature-based alignment fallback using ORB."""
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return img_color

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    if len(good) < 10:
        return img_color

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return img_color

    return cv2.warpPerspective(img_color, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REFLECT)


def create_luminosity_masks(img: np.ndarray) -> dict:
    """Create luminosity masks for targeted adjustments."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32) / 255.0

    masks = {
        "highlights": np.clip((L - 0.7) / 0.3, 0, 1),
        "midtones": 1.0 - np.abs(L - 0.5) * 2.0,
        "shadows": np.clip((0.3 - L) / 0.3, 0, 1),
        "bright_windows": np.clip((L - 0.85) / 0.15, 0, 1),
    }

    # Smooth masks to avoid harsh transitions
    for key in masks:
        masks[key] = cv2.GaussianBlur(masks[key], (51, 51), 0)

    return masks


def window_pull(merged: np.ndarray, darks: list[np.ndarray]) -> np.ndarray:
    """
    Dedicated window pull: recover blown-out window areas using darker exposures.
    This is the KEY differentiator for real estate HDR.
    """
    masks = create_luminosity_masks(merged)
    window_mask = masks["bright_windows"]

    if np.max(window_mask) < 0.05:
        logger.info("No blown highlights detected, skipping window pull")
        return merged

    # Find the darkest exposure for window recovery
    darkest = min(darks, key=lambda x: np.mean(cv2.cvtColor(x, cv2.COLOR_BGR2LAB)[:, :, 0]))

    # Blend the darkest exposure into the blown areas
    result = merged.astype(np.float32)
    dark_f = darkest.astype(np.float32)

    # Expand mask to 3 channels
    mask_3ch = np.stack([window_mask] * 3, axis=-1)

    # Weighted blend: use dark exposure in blown areas
    blend_strength = 0.75  # How much of the dark exposure to use
    result = result * (1 - mask_3ch * blend_strength) + dark_f * (mask_3ch * blend_strength)

    logger.info(f"Window pull applied, max mask intensity: {np.max(window_mask):.2f}")
    return np.clip(result, 0, 255).astype(np.uint8)


def exposure_fusion_pro(images: list[np.ndarray]) -> np.ndarray:
    """
    Professional exposure fusion with optimized weights for real estate.
    """
    merge = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=0.8,
        exposure_weight=1.0
    )
    fusion = merge.process(images)

    # Mertens outputs float [0,1] but can exceed range
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion


def enhance_local_contrast(img: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """Apply CLAHE on luminance for local contrast without color shifts."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def color_correction(img: np.ndarray) -> np.ndarray:
    """
    Correct color casts - critical for neutral white walls in real estate.
    Uses gray world assumption with safeguards.
    """
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


def process_job(job_id: str, image_urls: list[str], style: str = "natural",
                webhook_url: str = None, webhook_secret: str = None):
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

        # 2. Align images
        logger.info(f"[{job_id}] Aligning {len(images)} images...")
        aligned = align_images(images)

        # 3. Exposure fusion
        logger.info(f"[{job_id}] Running exposure fusion...")
        merged = exposure_fusion_pro(aligned)

        # 4. Window pull - recover blown-out windows using dark exposures
        logger.info(f"[{job_id}] Applying window pull...")
        merged = window_pull(merged, aligned)

        # 5. Color correction
        logger.info(f"[{job_id}] Correcting colors...")
        merged = color_correction(merged)

        # 6. Local contrast enhancement
        clip = 3.0 if style == "detailed" else 2.0
        logger.info(f"[{job_id}] Enhancing local contrast (clip={clip})...")
        merged = enhance_local_contrast(merged, clip_limit=clip)

        # 7. Light denoise
        logger.info(f"[{job_id}] Applying light denoise...")
        merged = denoise_light(merged)

        # 8. Smart sharpen
        logger.info(f"[{job_id}] Sharpening...")
        merged = smart_sharpen(merged, strength=0.35)

        # 9. Encode as high-quality JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 98]
        success, buffer = cv2.imencode(".jpg", merged, encode_params)
        if not success:
            raise RuntimeError("Failed to encode output image")

        result_b64 = base64.b64encode(buffer).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{result_b64}"
        size_kb = len(buffer) / 1024
        logger.info(f"[{job_id}] Output: {merged.shape}, {size_kb:.0f} KB")

        # Save locally for direct download
        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/{job_id}.jpg"
        with open(output_path, "wb") as f:
            f.write(buffer)

        # Send result via webhook
        if webhook_url:
            logger.info(f"[{job_id}] Sending result to webhook...")
            headers = {"Content-Type": "application/json"}
            if webhook_secret:
                headers["x-webhook-secret"] = webhook_secret
            payload = {
                "jobId": job_id,
                "status": "completed",
                "outputUrl": data_url,
            }
            resp = requests.post(webhook_url, json=payload, headers=headers, timeout=120)
            logger.info(f"[{job_id}] Webhook response: {resp.status_code}")

        return {"status": "completed", "output_path": output_path}

    except Exception as e:
        logger.error(f"[{job_id}] Processing failed: {e}", exc_info=True)

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
            except Exception:
                pass

        raise
