import os
import cv2
import numpy as np
import requests
import logging
import base64
import math

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
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    aligned = []

    for i, img in enumerate(images):
        if i == len(images) // 2:
            aligned.append(img)
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            # ECC sub-pixel alignment
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, img_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
            )
            aligned_img = cv2.warpAffine(
                img, warp_matrix, (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT_101
            )
            logger.info(f"[Worker] Image {i} aligned via ECC (euclidean)")
            aligned.append(aligned_img)

        except cv2.error:
            try:
                # ORB feature-matching fallback
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
                        img, H, (reference.shape[1], reference.shape[0]),
                        borderMode=cv2.BORDER_REFLECT_101
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


def detect_ghost_mask(images: list[np.ndarray]) -> np.ndarray:
    """Create a mask of moving objects (ghosting) across bracket images."""
    ref = images[len(images) // 2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

    ghost_mask = np.zeros(ref_gray.shape, dtype=np.float32)

    for i, img in enumerate(images):
        if i == len(images) // 2:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(ref_gray - gray)
        # Normalize by expected exposure difference
        diff = diff / (np.mean(diff) + 1e-6)
        ghost_mask = np.maximum(ghost_mask, diff)

    # Threshold and smooth
    _, binary = cv2.threshold(ghost_mask, 3.0, 1.0, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.GaussianBlur(binary, (21, 21), 0)

    return binary


def merge_hdr_with_ghost_removal(images: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Improved HDR merge with ghost removal and better tone mapping."""
    if weights is None:
        n = len(images)
        if n == 3:
            weights = [0.7, 1.0, 0.7]
        elif n == 5:
            weights = [0.5, 0.8, 1.0, 0.8, 0.5]
        else:
            weights = [1.0] * n

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    # Detect ghosting
    ghost_mask = detect_ghost_mask(images)
    ref_idx = len(images) // 2

    # Apply weights and ghost removal
    weighted_images = []
    for i, (img, w) in enumerate(zip(images, weights)):
        if i != ref_idx and np.any(ghost_mask > 0.1):
            # Blend ghost regions back to reference
            mask_3ch = np.stack([ghost_mask] * 3, axis=-1)
            img = (img.astype(np.float32) * (1 - mask_3ch) +
                   images[ref_idx].astype(np.float32) * mask_3ch).astype(np.uint8)
        weighted = cv2.convertScaleAbs(img, alpha=w)
        weighted_images.append(weighted)

    # Mertens exposure fusion with tuned parameters
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=0.8,
        exposure_weight=0.8,
    )
    fusion = merge_mertens.process(weighted_images)

    # Improved tone mapping: preserve highlights better
    fusion = np.clip(fusion, 0, 1)

    # Soft highlight rolloff (prevents blown highlights)
    highlight_threshold = 0.85
    highlight_mask = fusion > highlight_threshold
    fusion[highlight_mask] = highlight_threshold + (fusion[highlight_mask] - highlight_threshold) * 0.4

    # Shadow recovery (lift deep shadows gently)
    shadow_threshold = 0.08
    shadow_mask = fusion < shadow_threshold
    fusion[shadow_mask] = fusion[shadow_mask] * 1.3 + 0.01

    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE for local contrast enhancement in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def apply_color_correction(img: np.ndarray) -> np.ndarray:
    """LAB-space color correction for neutral whites + warmth."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Push a and b channels toward neutral (removes color casts)
    a = cv2.addWeighted(a, 0.92, np.full_like(a, 128), 0.08, 0)
    b = cv2.addWeighted(b, 0.92, np.full_like(b, 128), 0.08, 0)

    corrected = cv2.merge([l, a, b])
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def apply_sharpening(img: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Edge-aware unsharp mask with halo prevention."""
    blurred = cv2.GaussianBlur(img, (0, 0), 2.5)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

    # Clamp to prevent halos
    max_diff = 25
    diff = sharpened.astype(np.int16) - img.astype(np.int16)
    diff = np.clip(diff, -max_diff, max_diff)
    result = np.clip(img.astype(np.int16) + diff, 0, 255).astype(np.uint8)
    return result


def apply_denoising(img: np.ndarray) -> np.ndarray:
    """Light denoising - positional args only for OpenCV compatibility."""
    return cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 7, 21)


def correct_perspective(img: np.ndarray) -> np.ndarray:
    """Detect and correct vertical line perspective distortion (wall straightening)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=h * 0.15, maxLineGap=10)

    if lines is None or len(lines) < 4:
        logger.info("[Worker] Not enough lines detected for perspective correction")
        return img

    # Find near-vertical lines (within 15 degrees of vertical)
    vertical_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < h * 0.1:
            continue

        # Angle from vertical (0 = perfectly vertical)
        if abs(x2 - x1) < 1:
            angle = 0.0
        else:
            angle = math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1)))

        if angle < 15:
            # Signed angle: positive = leaning right, negative = leaning left
            signed_angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
            vertical_angles.append((signed_angle, length))

    if len(vertical_angles) < 3:
        logger.info("[Worker] Not enough vertical lines for correction")
        return img

    # Weight by line length
    total_weight = sum(length for _, length in vertical_angles)
    weighted_avg_angle = sum(angle * length for angle, length in vertical_angles) / total_weight

    # Only correct if tilt is noticeable but not extreme (0.3 to 5 degrees)
    if abs(weighted_avg_angle) < 0.3 or abs(weighted_avg_angle) > 5.0:
        logger.info(f"[Worker] Vertical tilt {weighted_avg_angle:.2f} degrees outside correction range")
        return img

    logger.info(f"[Worker] Correcting vertical tilt: {weighted_avg_angle:.2f} degrees ({len(vertical_angles)} lines)")

    # Apply perspective correction using keystone transform
    cx, cy = w / 2, h / 2
    angle_rad = math.radians(weighted_avg_angle)

    # Subtle keystone: shift top edge proportional to the tilt
    shift = math.tan(angle_rad) * h * 0.5

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [shift, 0], [w + shift, 0],
        [w - shift, h], [-shift, h]
    ])

    # Use perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)

    logger.info("[Worker] Perspective correction applied")
    return corrected


def process_job(payload: dict):
    """Main HDR processing pipeline."""
    job_id = payload.get("jobId") or payload.get("job_id", "")
    input_urls = payload.get("inputUrls") or payload.get("input_urls", [])
    webhook_url = payload.get("webhookUrl") or payload.get("webhook_url")
    webhook_secret = payload.get("webhookSecret") or payload.get("webhook_secret")
    weights = payload.get("weights")
    style = payload.get("style", "natural")

    logger.info(f"[Worker] Processing job {job_id} with {len(input_urls)} images, style={style}")

    try:
        # 1. Download all bracket images
        images = [download_image(url) for url in input_urls]
        logger.info(f"[Worker] Downloaded {len(images)} images")

        # 2. Align images (ECC with ORB fallback)
        aligned = align_images(images)
        logger.info(f"[Worker] Alignment complete")

        # 3. HDR merge with ghost removal and improved tone mapping
        merged = merge_hdr_with_ghost_removal(aligned, weights)
        logger.info(f"[Worker] HDR fusion complete: {merged.shape}")

        # 4. Perspective/wall straightening
        merged = correct_perspective(merged)

        # 5. CLAHE local contrast (slightly gentler to avoid over-processing)
        enhanced = apply_clahe(merged, clip_limit=2.0, grid_size=8)
        logger.info(f"[Worker] CLAHE applied")

        # 6. Color correction (stronger neutralization)
        enhanced = apply_color_correction(enhanced)
        logger.info(f"[Worker] Color correction applied")

        # 7. Light denoising
        enhanced = apply_denoising(enhanced)
        logger.info(f"[Worker] Denoising applied")

        # 8. Edge-aware sharpening (with halo prevention)
        enhanced = apply_sharpening(enhanced, strength=0.4)
        logger.info(f"[Worker] Sharpening applied")

        # 9. Encode to JPEG at 97% quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 97]
        success, buffer = cv2.imencode(".jpg", enhanced, encode_params)
        if not success:
            raise ValueError("Failed to encode result image")

        result_bytes = buffer.tobytes()
        logger.info(f"[Worker] Encoded result: {len(result_bytes)} bytes ({len(result_bytes)/1024:.1f}KB)")

        # 10. Save to outputs directory for polling fallback
        out_dir = os.getenv("OUTPUT_DIR", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{job_id}.jpg")
        with open(out_path, "wb") as f:
            f.write(result_bytes)
        logger.info(f"[Worker] Saved to {out_path}")

        # 11. Send result via webhook
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
                requests.post(webhook_url, json=webhook_payload, headers=headers, timeout=30)
            except Exception:
                pass

        raise
