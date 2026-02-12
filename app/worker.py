"""
Snapfix HDR Worker v5.2
Professional real estate HDR merge with OpenCV.
"""

import os
import io
import cv2
import json
import base64
import logging
import tempfile
import requests
import numpy as np
from redis import Redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hdr-worker")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_conn = Redis.from_url(REDIS_URL)

RESULTS_PREFIX = "hdr_result:"
STATUS_PREFIX = "hdr_status:"

# ========== QUALITY SETTINGS ==========
FINAL_JPEG_QUALITY = 97       # Final output quality (professional grade)
WEBHOOK_JPEG_QUALITY = 85     # Transfer quality (bypass Railway 120s timeout)
# ======================================


def download_image(url: str) -> np.ndarray:
    """Download image from URL and return as BGR numpy array."""
    logger.info(f"Downloading: {url[:100]}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from URL")
    logger.info(f"Downloaded image: {img.shape[1]}x{img.shape[0]}")
    return img


def detect_scene(images: list) -> str:
    """Detect if scene is interior or exterior based on image characteristics."""
    ref = images[1]  # Normal exposure
    hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    
    # Check for sky-like regions (high value, low-mid saturation, blue-ish hue)
    h, s, v = cv2.split(hsv)
    sky_mask = (h > 90) & (h < 130) & (s > 30) & (v > 150)
    sky_ratio = np.sum(sky_mask) / sky_mask.size
    
    # Check for green vegetation
    green_mask = (h > 35) & (h < 85) & (s > 40) & (v > 50)
    green_ratio = np.sum(green_mask) / green_mask.size
    
    if sky_ratio > 0.08 or green_ratio > 0.15:
        logger.info(f"Scene detected: EXTERIOR (sky={sky_ratio:.3f}, green={green_ratio:.3f})")
        return "exterior"
    else:
        logger.info(f"Scene detected: INTERIOR (sky={sky_ratio:.3f}, green={green_ratio:.3f})")
        return "interior"


def align_images(images: list) -> list:
    """Sub-pixel alignment using ECC (Enhanced Correlation Coefficient)."""
    logger.info("Aligning images...")
    ref = images[1]  # Normal exposure as reference
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    
    aligned = [None, ref, None]
    warp_mode = cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    
    for i in [0, 2]:
        try:
            img_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            _, warp_matrix = cv2.findTransformECC(ref_gray, img_gray, warp_matrix, warp_mode, criteria)
            h, w = ref.shape[:2]
            aligned[i] = cv2.warpAffine(
                images[i], warp_matrix, (w, h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
            logger.info(f"Image {i} aligned (shift: dx={warp_matrix[0,2]:.2f}, dy={warp_matrix[1,2]:.2f})")
        except cv2.error as e:
            logger.warning(f"ECC alignment failed for image {i}, using original: {e}")
            aligned[i] = images[i]
    
    return aligned


def remove_ghosts(images: list, merged: np.ndarray) -> np.ndarray:
    """Remove ghosting artifacts from moving objects."""
    ref_gray = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    for i in [0, 2]:
        img_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(ref_gray - img_gray)
        ghost_mask = (diff > 30).astype(np.float32)
        ghost_mask = cv2.GaussianBlur(ghost_mask, (15, 15), 0)
        
        for c in range(3):
            merged[:, :, c] = (
                merged[:, :, c] * (1 - ghost_mask) +
                images[1][:, :, c] * ghost_mask
            ).astype(np.uint8)
    
    return merged


def exposure_fusion(images: list) -> np.ndarray:
    """Mertens exposure fusion — the core HDR merge."""
    logger.info("Running Mertens exposure fusion...")
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0,
    )
    fusion = merge_mertens.process(images)
    # Clip and convert to 8-bit
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion


def apply_tone_mapping(img: np.ndarray, scene: str) -> np.ndarray:
    """LAB-only tone mapping to preserve original colors."""
    logger.info("Applying LAB tone mapping...")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    
    # Gamma correction (0.91 — subtle lift)
    gamma = 0.91
    l_norm = l_channel / 255.0
    l_norm = np.power(l_norm, gamma)
    
    # S-curve for contrast (strength 0.50)
    s_strength = 0.50
    l_scurve = l_norm - s_strength * (l_norm - 0.5) * l_norm * (1 - l_norm) * 4
    l_scurve = np.clip(l_scurve, 0, 1)
    
    lab[:, :, 0] = l_scurve * 255.0
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def apply_white_balance(img: np.ndarray) -> np.ndarray:
    """Balanced AWB with warm-clamp (red boost capped at 1.03)."""
    logger.info("Applying white balance...")
    strength = 0.12
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])
    
    # Shift toward neutral (128 is neutral in LAB)
    lab[:, :, 1] = lab[:, :, 1] - strength * (avg_a - 128)
    lab[:, :, 2] = lab[:, :, 2] - strength * (avg_b - 128)
    
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Warm-clamp: cap red channel boost at 1.03x
    b, g, r = cv2.split(result)
    _, g_orig, _ = cv2.split(img)
    r_ratio = np.mean(r.astype(np.float32)) / max(np.mean(g.astype(np.float32)), 1)
    r_orig_ratio = np.mean(cv2.split(img)[2].astype(np.float32)) / max(np.mean(g_orig.astype(np.float32)), 1)
    
    if r_ratio > r_orig_ratio * 1.03:
        scale = (r_orig_ratio * 1.03) / max(r_ratio, 0.01)
        r = np.clip(r.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        result = cv2.merge([b, g, r])
    
    return result


def recover_shadows(img: np.ndarray, dark_img: np.ndarray) -> np.ndarray:
    """Enhanced shadow recovery (0.45 threshold, 0.35 strength)."""
    logger.info("Recovering shadows...")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0] / 255.0
    
    threshold = 0.45
    strength = 0.35
    
    shadow_mask = np.clip((threshold - l_channel) / threshold, 0, 1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (31, 31), 0)
    
    # Lift shadows
    lift = shadow_mask * strength * (1 - l_channel)
    l_channel = l_channel + lift
    
    lab[:, :, 0] = np.clip(l_channel * 255, 0, 255)
    lab = lab.astype(np.uint8)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def preserve_surface_tones(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Surface tone preservation using pre-fusion LAB signatures."""
    logger.info("Preserving surface tones...")
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Identify large uniform surfaces (walls, floors, ceilings)
    l_ref = lab_ref[:, :, 0]
    threshold = 0.80  # 80% threshold
    
    # Low-texture regions = surfaces
    grad_x = cv2.Sobel(l_ref, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(l_ref, cv2.CV_32F, 0, 1, ksize=5)
    gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    max_grad = np.max(gradient_mag) if np.max(gradient_mag) > 0 else 1
    surface_mask = (gradient_mag / max_grad < (1 - threshold)).astype(np.float32)
    surface_mask = cv2.GaussianBlur(surface_mask, (21, 21), 0)
    
    # Blend color channels toward reference on surfaces
    for c in [1, 2]:  # a and b channels only
        lab_img[:, :, c] = (
            lab_img[:, :, c] * (1 - surface_mask * 0.6) +
            lab_ref[:, :, c] * (surface_mask * 0.6)
        )
    
    lab_img = np.clip(lab_img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)


def correct_perspective(img: np.ndarray) -> np.ndarray:
    """Perspective correction using Hough line detection."""
    logger.info("Correcting perspective...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=img.shape[0] // 4, maxLineGap=20)
    
    if lines is None or len(lines) < 3:
        logger.info("Not enough lines detected, skipping perspective correction")
        return img
    
    # Find near-vertical lines
    vertical_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Near-vertical: close to ±90°
        if 70 < abs(angle) < 110:
            vertical_angles.append(angle)
    
    if len(vertical_angles) < 2:
        logger.info("Not enough vertical lines, skipping correction")
        return img
    
    # Median deviation from true vertical
    median_angle = np.median(vertical_angles)
    correction = 90 - abs(median_angle) if median_angle > 0 else -(90 - abs(median_angle))
    
    if abs(correction) < 0.3:
        logger.info(f"Negligible correction ({correction:.2f}°), skipping")
        return img
    
    if abs(correction) > 5:
        logger.info(f"Correction too large ({correction:.2f}°), capping at 5°")
        correction = np.clip(correction, -5, 5)
    
    logger.info(f"Applying keystone correction: {correction:.2f}°")
    h, w = img.shape[:2]
    
    # Simple rotation for small corrections
    M = cv2.getRotationMatrix2D((w / 2, h / 2), correction, 1.0)
    result = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REFLECT)
    return result


def apply_edge_aware_contrast(img: np.ndarray) -> np.ndarray:
    """Edge-aware local contrast enhancement."""
    logger.info("Applying edge-aware contrast...")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # CLAHE with conservative settings
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    # Blend 40% enhanced with 60% original to keep it subtle
    l_blended = cv2.addWeighted(l_channel, 0.6, l_enhanced, 0.4, 0)
    lab[:, :, 0] = l_blended
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_denoising(img: np.ndarray, scene: str) -> np.ndarray:
    """Adaptive denoising — stronger for interiors."""
    logger.info(f"Applying denoising (scene={scene})...")
    h_lum = 2 if scene == "exterior" else 3
    h_color = 2
    # MUST use positional args for OpenCV compatibility on Railway
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h_lum, h_color, 7, 21)
    return denoised


def apply_lens_correction(img: np.ndarray) -> np.ndarray:
    """Lens distortion correction — removes barrel/pincushion distortion."""
    logger.info("Applying lens correction...")
    h, w = img.shape[:2]
    
    # Estimate a mild barrel distortion correction
    # These are conservative values that work for typical wide-angle real estate lenses
    focal_length = w * 0.8  # Approximate focal length
    cx, cy = w / 2, h / 2
    
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Mild barrel distortion correction coefficients
    dist_coeffs = np.array([-0.08, 0.02, 0, 0, 0], dtype=np.float64)
    
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 0.3, (w, h)
    )
    
    corrected = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # Crop to ROI if valid
    x, y, rw, rh = roi
    if rw > w * 0.9 and rh > h * 0.9:
        corrected = corrected[y:y+rh, x:x+rw]
        corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    return corrected


def apply_dehaze(img: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """Dark channel prior dehazing for clarity."""
    logger.info(f"Applying dehaze (strength={strength})...")
    img_f = img.astype(np.float64) / 255.0
    
    # Dark channel
    kernel_size = 15
    dark = np.min(img_f, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark_channel = cv2.erode(dark, kernel)
    
    # Atmospheric light estimate
    flat_dark = dark_channel.flatten()
    num_pixels = max(int(flat_dark.size * 0.001), 1)
    indices = np.argsort(flat_dark)[-num_pixels:]
    
    atm_light = np.zeros(3)
    for c in range(3):
        flat_c = img_f[:, :, c].flatten()
        atm_light[c] = np.max(flat_c[indices])
    atm_light = np.clip(atm_light, 0.5, 1.0)
    
    # Transmission estimate
    norm = img_f / atm_light[np.newaxis, np.newaxis, :]
    norm_dark = np.min(norm, axis=2)
    kernel_t = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    transmission = 1 - strength * cv2.erode(norm_dark, kernel_t)
    transmission = np.clip(transmission, 0.1, 1.0)
    
    # Guided filter refinement (simplified)
    transmission = cv2.GaussianBlur(transmission, (31, 31), 0)
    
    # Recover scene
    result = np.zeros_like(img_f)
    for c in range(3):
        result[:, :, c] = (img_f[:, :, c] - atm_light[c]) / np.maximum(transmission, 0.1) + atm_light[c]
    
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


def apply_highlight_recovery(img: np.ndarray, dark_img: np.ndarray) -> np.ndarray:
    """Recover blown highlights using the dark exposure."""
    logger.info("Recovering highlights...")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_dark = cv2.cvtColor(dark_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    l_channel = lab[:, :, 0] / 255.0
    
    # Highlight mask: pixels above 85% brightness
    highlight_threshold = 0.85
    highlight_mask = np.clip((l_channel - highlight_threshold) / (1 - highlight_threshold), 0, 1)
    highlight_mask = cv2.GaussianBlur(highlight_mask, (21, 21), 0)
    
    # Blend from dark exposure in highlight regions
    blend_strength = 0.7
    for c in range(3):
        lab[:, :, c] = (
            lab[:, :, c] * (1 - highlight_mask * blend_strength) +
            lab_dark[:, :, c] * (highlight_mask * blend_strength)
        )
    
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_sharpening(img: np.ndarray) -> np.ndarray:
    """Unsharp mask for architectural sharpness."""
    logger.info("Applying sharpening...")
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened = cv2.addWeighted(img, 1.3, gaussian, -0.3, 0)
    return sharpened


def process_job(payload: dict):
    """Main HDR processing pipeline."""
    job_id = payload["job_id"]
    input_urls = payload["input_urls"]
    style = payload.get("style", "natural")
    webhook_url = payload.get("webhook_url")
    webhook_secret = payload.get("webhook_secret")

    logger.info(f"[Worker] Starting job {job_id}, style={style}, inputs={len(input_urls)}")

    try:
        # 1. Download all input images
        images = []
        for url in input_urls:
            images.append(download_image(url))
        
        logger.info(f"[Worker] Downloaded {len(images)} images")
        
        # Store reference for surface tone preservation
        reference_normal = images[1].copy()
        
        # 2. Scene detection
        scene = detect_scene(images)
        
        # 3. Sub-pixel alignment
        aligned = align_images(images)
        
        # 4. Lens correction on all inputs
        aligned = [apply_lens_correction(img) for img in aligned]
        
        # 5. Exposure fusion (core HDR merge)
        merged = exposure_fusion(aligned)
        
        # 6. Ghost removal
        merged = remove_ghosts(aligned, merged)
        
        # 7. Highlight recovery from dark exposure
        merged = apply_highlight_recovery(merged, aligned[0])
        
        # 8. LAB tone mapping
        merged = apply_tone_mapping(merged, scene)
        
        # 9. White balance
        merged = apply_white_balance(merged)
        
        # 10. Shadow recovery
        merged = recover_shadows(merged, aligned[0])
        
        # 11. Dehaze
        merged = apply_dehaze(merged, strength=0.2)
        
        # 12. Surface tone preservation
        merged = preserve_surface_tones(merged, reference_normal)
        
        # 13. Edge-aware contrast
        merged = apply_edge_aware_contrast(merged)
        
        # 14. Perspective correction
        merged = correct_perspective(merged)
        
        # 15. Adaptive denoising
        merged = apply_denoising(merged, scene)
        
        # 16. Final sharpening
        merged = apply_sharpening(merged)

        logger.info(f"[Worker] Processing complete. Output: {merged.shape[1]}x{merged.shape[0]}")

        # Encode result
        encode_params_final = [cv2.IMWRITE_JPEG_QUALITY, FINAL_JPEG_QUALITY]
        _, final_buffer = cv2.imencode(".jpg", merged, encode_params_final)
        final_bytes = final_buffer.tobytes()
        
        logger.info(f"[Worker] Final JPEG: {len(final_bytes) / 1024:.1f} KB at {FINAL_JPEG_QUALITY}% quality")

        # Send via webhook if configured
        if webhook_url:
            logger.info(f"[Worker] Sending webhook to: {webhook_url}")
            
            # Re-encode at lower quality for transfer
            encode_params_webhook = [cv2.IMWRITE_JPEG_QUALITY, WEBHOOK_JPEG_QUALITY]
            _, webhook_buffer = cv2.imencode(".jpg", merged, encode_params_webhook)
            webhook_bytes = webhook_buffer.tobytes()
            
            result_b64 = base64.b64encode(webhook_bytes).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{result_b64}"
            
            logger.info(f"[Worker] Webhook payload: {len(webhook_bytes) / 1024:.1f} KB "
                        f"(transfer at {WEBHOOK_JPEG_QUALITY}%)")

            webhook_payload = {
                "jobId": job_id,
                "job_id": job_id,
                "status": "completed",
                "resultImage": data_url,
                "result_image": data_url,
                "width": merged.shape[1],
                "height": merged.shape[0],
            }

            headers = {"Content-Type": "application/json"}
            if webhook_secret:
                headers["X-Webhook-Secret"] = webhook_secret

            try:
                resp = requests.post(webhook_url, json=webhook_payload, headers=headers, timeout=110)
                logger.info(f"[Worker] Webhook response: {resp.status_code}")
                if resp.status_code >= 400:
                    logger.error(f"[Worker] Webhook failed: {resp.text[:300]}")
            except Exception as e:
                logger.error(f"[Worker] Webhook error: {e}")

        # Store result in Redis for polling fallback
        # Store as a temporary signed URL or just mark as completed
        redis_conn.set(f"{STATUS_PREFIX}{job_id}", "completed", ex=3600)
        # For polling fallback, we'd need to store the image somewhere accessible
        # The webhook is the primary delivery mechanism
        
        logger.info(f"[Worker] Job {job_id} completed successfully")
        return {"status": "completed", "job_id": job_id}

    except Exception as e:
        logger.error(f"[Worker] Job {job_id} FAILED: {e}", exc_info=True)
        redis_conn.set(f"{STATUS_PREFIX}{job_id}", "failed", ex=3600)
        redis_conn.set(f"{RESULTS_PREFIX}{job_id}:error", str(e), ex=3600)

        # Try to send failure webhook
        if webhook_url:
            try:
                headers = {"Content-Type": "application/json"}
                if webhook_secret:
                    headers["X-Webhook-Secret"] = webhook_secret
                requests.post(webhook_url, json={
                    "jobId": job_id,
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(e),
                }, headers=headers, timeout=30)
            except Exception:
                pass

        raise
