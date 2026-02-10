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
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, img_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
            )
            aligned_img = cv2.warpAffine(
                img, warp_matrix, (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT_101
            )
            logger.info(f"[Worker] Image {i} aligned via ECC")
            aligned.append(aligned_img)

        except cv2.error:
            try:
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
                logger.warning(f"[Worker] Image {i} alignment failed: {e}")
                aligned.append(img)

    return aligned


def detect_ghost_mask(images: list[np.ndarray]) -> np.ndarray:
    """Create a mask of moving objects across bracket images."""
    ref = images[len(images) // 2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ghost_mask = np.zeros(ref_gray.shape, dtype=np.float32)

    for i, img in enumerate(images):
        if i == len(images) // 2:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(ref_gray - gray)
        diff = diff / (np.mean(diff) + 1e-6)
        ghost_mask = np.maximum(ghost_mask, diff)

    _, binary = cv2.threshold(ghost_mask, 3.0, 1.0, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.GaussianBlur(binary, (21, 21), 0)
    return binary


def merge_hdr_with_ghost_removal(images: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """HDR merge with ghost removal and professional tone mapping."""
    if weights is None:
        n = len(images)
        if n == 3:
            weights = [0.8, 1.0, 0.8]
        elif n == 5:
            weights = [0.5, 0.8, 1.0, 0.8, 0.5]
        else:
            weights = [1.0] * n

    total = sum(weights)
    weights = [w / total for w in weights]

    ghost_mask = detect_ghost_mask(images)
    ref_idx = len(images) // 2

    processed_images = []
    for i, (img, w) in enumerate(zip(images, weights)):
        if i != ref_idx and np.any(ghost_mask > 0.1):
            mask_3ch = np.stack([ghost_mask] * 3, axis=-1)
            img = (img.astype(np.float32) * (1 - mask_3ch) +
                   images[ref_idx].astype(np.float32) * mask_3ch).astype(np.uint8)
        processed_images.append(img)

    # Mertens exposure fusion
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=0.0,
    )
    fusion = merge_mertens.process(processed_images)
    fusion = np.clip(fusion, 0, 1)

    # ---- PROFESSIONAL TONE MAPPING ----

    # 1. S-curve tone mapping for cinematic contrast
    fusion = apply_s_curve(fusion)

    # 2. Soft highlight rolloff (prevents blown highlights)
    highlight_threshold = 0.92
    mask = fusion > highlight_threshold
    fusion[mask] = highlight_threshold + (fusion[mask] - highlight_threshold) * 0.3

    # 3. Shadow lift (open up dark areas naturally)
    shadow_threshold = 0.12
    shadow_mask = fusion < shadow_threshold
    # Quadratic lift — gentle at the bottom, stronger as it approaches threshold
    fusion[shadow_mask] = fusion[shadow_mask] + (shadow_threshold - fusion[shadow_mask]) * 0.35

    # 4. Gamma correction for brightness punch
    gamma = 0.88  # < 1 brightens
    fusion = np.power(np.clip(fusion, 0, 1), gamma)

    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion


def apply_s_curve(img: np.ndarray) -> np.ndarray:
    """Apply an S-curve for professional contrast (operates on 0-1 float image)."""
    # Attempt a smooth S-curve using a sigmoid-like function
    # This adds contrast in the midtones while preserving shadows and highlights
    midpoint = 0.5
    strength = 0.6  # How aggressive the S-curve is (0 = none, 1 = max)

    # Soft S-curve: blend between linear and sigmoid
    sigmoid = 1.0 / (1.0 + np.exp(-12.0 * (img - midpoint)))
    result = img * (1.0 - strength) + sigmoid * strength
    return np.clip(result, 0, 1)


def apply_clahe_pro(img: np.ndarray) -> np.ndarray:
    """Professional CLAHE with separate treatment for shadows, mids, highlights."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Strong CLAHE for local contrast
    clahe_strong = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_strong = clahe_strong.apply(l)

    # Gentle CLAHE to avoid over-processing bright areas
    clahe_gentle = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    l_gentle = clahe_gentle.apply(l)

    # Blend: use strong CLAHE in darks/mids, gentle in highlights
    l_float = l.astype(np.float32) / 255.0
    highlight_weight = np.clip((l_float - 0.6) / 0.4, 0, 1)  # 0 in shadows, 1 in highlights

    l_blended = (l_strong.astype(np.float32) * (1 - highlight_weight) +
                 l_gentle.astype(np.float32) * highlight_weight)
    l_blended = np.clip(l_blended, 0, 255).astype(np.uint8)

    lab_enhanced = cv2.merge([l_blended, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_color_correction_pro(img: np.ndarray) -> np.ndarray:
    """Professional LAB-space color correction with strong neutral white push and vibrance."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Strong push toward neutral (removes yellow/green/blue casts)
    # 20% correction toward neutral center (128)
    a = cv2.addWeighted(a, 0.80, np.full_like(a, 128), 0.20, 0)
    b = cv2.addWeighted(b, 0.80, np.full_like(b, 128), 0.20, 0)

    # Selective saturation boost in midtones only (vibrance effect)
    # Don't boost already-saturated pixels
    a_float = a.astype(np.float32) - 128.0
    b_float = b.astype(np.float32) - 128.0
    saturation = np.sqrt(a_float**2 + b_float**2)

    # Low saturation pixels get boosted more (vibrance)
    boost_factor = 1.0 + 0.15 * np.clip(1.0 - saturation / 40.0, 0, 1)
    a_boosted = (a_float * boost_factor + 128.0).astype(np.uint8)
    b_boosted = (b_float * boost_factor + 128.0).astype(np.uint8)

    corrected = cv2.merge([l, np.clip(a_boosted, 0, 255).astype(np.uint8),
                           np.clip(b_boosted, 0, 255).astype(np.uint8)])
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def apply_white_balance(img: np.ndarray) -> np.ndarray:
    """Auto white balance using the gray world assumption with limits."""
    result = img.astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    # Limit correction to prevent extreme shifts
    scale_b = np.clip(avg_all / (avg_b + 1e-6), 0.8, 1.2)
    scale_g = np.clip(avg_all / (avg_g + 1e-6), 0.8, 1.2)
    scale_r = np.clip(avg_all / (avg_r + 1e-6), 0.8, 1.2)

    result[:, :, 0] *= scale_b
    result[:, :, 1] *= scale_g
    result[:, :, 2] *= scale_r

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_sharpening_pro(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Multi-scale edge-aware sharpening with halo prevention."""
    # Fine detail sharpening
    blur_fine = cv2.GaussianBlur(img, (0, 0), 1.0)
    detail_fine = cv2.addWeighted(img, 1.0 + strength * 0.6, blur_fine, -strength * 0.6, 0)

    # Medium structure sharpening
    blur_med = cv2.GaussianBlur(img, (0, 0), 3.0)
    detail_med = cv2.addWeighted(detail_fine, 1.0 + strength * 0.4, blur_med, -strength * 0.4, 0)

    # Halo prevention: clamp deviation from original
    max_diff = 20
    diff = detail_med.astype(np.int16) - img.astype(np.int16)
    diff = np.clip(diff, -max_diff, max_diff)
    result = np.clip(img.astype(np.int16) + diff, 0, 255).astype(np.uint8)
    return result


def apply_denoising(img: np.ndarray) -> np.ndarray:
    """Light denoising — positional args for OpenCV compatibility."""
    return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)


def correct_perspective(img: np.ndarray) -> np.ndarray:
    """Detect and correct vertical line perspective distortion (wall straightening)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=h * 0.15, maxLineGap=10)

    if lines is None or len(lines) < 4:
        logger.info("[Worker] Not enough lines for perspective correction")
        return img

    vertical_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < h * 0.1:
            continue

        if abs(x2 - x1) < 1:
            angle = 0.0
        else:
            angle = math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1)))

        if angle < 15:
            signed_angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
            vertical_angles.append((signed_angle, length))

    if len(vertical_angles) < 3:
        logger.info("[Worker] Not enough vertical lines for correction")
        return img

    total_weight = sum(length for _, length in vertical_angles)
    weighted_avg_angle = sum(angle * length for angle, length in vertical_angles) / total_weight

    if abs(weighted_avg_angle) < 0.3 or abs(weighted_avg_angle) > 5.0:
        logger.info(f"[Worker] Tilt {weighted_avg_angle:.2f}° outside correction range")
        return img

    logger.info(f"[Worker] Correcting tilt: {weighted_avg_angle:.2f}° ({len(vertical_angles)} lines)")

    angle_rad = math.radians(weighted_avg_angle)
    shift = math.tan(angle_rad) * h * 0.5

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [shift, 0], [w + shift, 0],
        [w - shift, h], [-shift, h]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected = cv2.warpPerspective(img, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
    logger.info("[Worker] Perspective correction applied")
    return corrected


def apply_final_brightness_contrast(img: np.ndarray) -> np.ndarray:
    """Final brightness/contrast pass to match professional real estate look."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Check average luminance and boost if image is too dark
    avg_l = np.mean(l)
    logger.info(f"[Worker] Average luminance: {avg_l:.1f}/255")

    if avg_l < 120:
        # Dark image — lift luminance
        target = 135
        alpha = min((target / (avg_l + 1e-6)), 1.4)
        beta = 10
        l = cv2.convertScaleAbs(l, alpha=alpha, beta=beta)
        logger.info(f"[Worker] Brightness boost applied: alpha={alpha:.2f}, beta={beta}")
    elif avg_l > 180:
        # Over-bright — gentle pull down
        alpha = 0.92
        l = cv2.convertScaleAbs(l, alpha=alpha, beta=-5)
        logger.info("[Worker] Brightness reduced slightly")

    lab_adjusted = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)


def process_job(payload: dict):
    """Main HDR processing pipeline — professional real estate quality."""
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
        logger.info("[Worker] Alignment complete")

        # 3. HDR merge with ghost removal + S-curve tone mapping
        merged = merge_hdr_with_ghost_removal(aligned, weights)
        logger.info(f"[Worker] HDR fusion complete: {merged.shape}")

        # 4. Perspective/wall straightening
        merged = correct_perspective(merged)

        # 5. Auto white balance (gray world)
        enhanced = apply_white_balance(merged)
        logger.info("[Worker] White balance applied")

        # 6. Professional CLAHE (shadow/highlight adaptive)
        enhanced = apply_clahe_pro(enhanced)
        logger.info("[Worker] Pro CLAHE applied")

        # 7. Strong color correction + vibrance
        enhanced = apply_color_correction_pro(enhanced)
        logger.info("[Worker] Pro color correction applied")

        # 8. Final brightness/contrast adjustment
        enhanced = apply_final_brightness_contrast(enhanced)
        logger.info("[Worker] Brightness/contrast finalized")

        # 9. Light denoising
        enhanced = apply_denoising(enhanced)
        logger.info("[Worker] Denoising applied")

        # 10. Multi-scale edge-aware sharpening
        enhanced = apply_sharpening_pro(enhanced, strength=0.5)
        logger.info("[Worker] Pro sharpening applied")

        # 11. Encode to JPEG at 97% quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 97]
        success, buffer = cv2.imencode(".jpg", enhanced, encode_params)
        if not success:
            raise ValueError("Failed to encode result image")

        result_bytes = buffer.tobytes()
        logger.info(f"[Worker] Encoded result: {len(result_bytes)} bytes ({len(result_bytes)/1024:.1f}KB)")

        # 12. Save to outputs directory
        out_dir = os.getenv("OUTPUT_DIR", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{job_id}.jpg")
        with open(out_path, "wb") as f:
            f.write(result_bytes)
        logger.info(f"[Worker] Saved to {out_path}")

        # 13. Send result via webhook
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

            logger.info(f"[Worker] Sending webhook to {webhook_url[:60]}... payload: {len(result_b64)//1024}KB")

            try:
                resp = requests.post(webhook_url, json=webhook_payload, headers=headers, timeout=120)
                logger.info(f"[Worker] Webhook response: {resp.status_code} - {resp.text[:200]}")
            except Exception as e:
                logger.error(f"[Worker] Webhook delivery failed: {e}")
        else:
            logger.info("[Worker] No webhook URL, result via polling only")

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
                requests.post(webhook_url, json=fail_payload, headers=headers, timeout=30)
            except Exception:
                pass

        raise
