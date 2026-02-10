"""
Snapfix HDR Worker – Professional Real Estate HDR Processing Engine
v5.2 – Balanced warmth preservation + natural wood tones + webhook compression
"""

import io
import os
import time
import json
import traceback
import requests
import numpy as np
import cv2
from redis import Redis
from rq import Worker, Queue

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
JPEG_QUALITY = 97


# ─── Scene Detection ────────────────────────────────────────────────

def detect_scene_type(images):
    img = images[len(images) // 2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    top_portion = h[:img.shape[0] // 3, :]
    top_s = s[:img.shape[0] // 3, :]
    top_v = v[:img.shape[0] // 3, :]

    sky_mask = ((top_portion > 85) & (top_portion < 135) &
                (top_s > 30) & (top_v > 100))
    sky_ratio = np.sum(sky_mask) / sky_mask.size
    avg_top_brightness = np.mean(top_v)

    green_mask = ((h > 30) & (h < 85) & (s > 30))
    green_ratio = np.sum(green_mask) / green_mask.size

    is_exterior = (sky_ratio > 0.08 or avg_top_brightness > 170 or green_ratio > 0.15)
    scene = "exterior" if is_exterior else "interior"
    print(f"[Scene Detection] Type: {scene} | Sky: {sky_ratio:.2%} | "
          f"Top brightness: {avg_top_brightness:.0f} | Green: {green_ratio:.2%}")
    return scene


# ─── Alignment ───────────────────────────────────────────────────────

def align_images_ecc(images):
    if len(images) < 2:
        return images

    ref = images[len(images) // 2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    aligned = []

    for i, img in enumerate(images):
        if i == len(images) // 2:
            aligned.append(ref)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
            )
            result = cv2.warpAffine(
                img, warp_matrix, (ref.shape[1], ref.shape[0]),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT101
            )
            print(f"  [Align] Frame {i}: ECC success")
            aligned.append(result)
        except cv2.error:
            print(f"  [Align] Frame {i}: ECC failed, trying ORB fallback")
            try:
                orb = cv2.ORB_create(5000)
                kp1, des1 = orb.detectAndCompute(ref_gray, None)
                kp2, des2 = orb.detectAndCompute(gray, None)

                if des1 is not None and des2 is not None:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)[:100]

                    if len(matches) >= 10:
                        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
                        M, _ = cv2.estimateAffinePartial2D(pts2, pts1)
                        if M is not None:
                            result = cv2.warpAffine(
                                img, M, (ref.shape[1], ref.shape[0]),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REFLECT101
                            )
                            print(f"  [Align] Frame {i}: ORB success ({len(matches)} matches)")
                            aligned.append(result)
                            continue

                print(f"  [Align] Frame {i}: ORB fallback failed, using unaligned")
                aligned.append(img)
            except Exception as e:
                print(f"  [Align] Frame {i}: ORB error: {e}, using unaligned")
                aligned.append(img)

    return aligned


# ─── Ghost Removal ───────────────────────────────────────────────────

def remove_ghosts(images):
    if len(images) < 2:
        return images

    ref = images[len(images) // 2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result = []

    for i, img in enumerate(images):
        if i == len(images) // 2:
            result.append(img)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(ref_gray - gray)
        _, ghost_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        ghost_mask = ghost_mask.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        ghost_mask = cv2.morphologyEx(ghost_mask, cv2.MORPH_CLOSE, kernel)
        ghost_mask = cv2.GaussianBlur(ghost_mask, (21, 21), 0)

        ghost_ratio = np.sum(ghost_mask > 127) / ghost_mask.size
        if ghost_ratio > 0.005:
            mask_3ch = cv2.merge([ghost_mask, ghost_mask, ghost_mask]) / 255.0
            blended = (ref * mask_3ch + img * (1 - mask_3ch)).astype(np.uint8)
            print(f"  [Ghost] Frame {i}: masked {ghost_ratio:.2%}")
            result.append(blended)
        else:
            result.append(img)

    return result


# ─── Exposure Fusion ─────────────────────────────────────────────────

def exposure_fusion(images, scene_type="interior"):
    merge = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0
    )

    n = len(images)
    mid = n // 2

    if scene_type == "exterior":
        weights = np.ones(n, dtype=np.float32)
        weights[0] = 1.4
        weights[mid] = 1.0
        weights[-1] = 0.7
    else:
        weights = np.ones(n, dtype=np.float32)
        weights[mid] = 1.3
        weights[-1] = 1.1
        weights[0] = 0.8

    weighted = []
    for i, img in enumerate(images):
        w = weights[i] if i < len(weights) else 1.0
        adjusted = np.clip(img.astype(np.float32) * w, 0, 255).astype(np.uint8)
        weighted.append(adjusted)

    fusion = merge.process(weighted)
    result = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return result


# ─── LAB-Only Tone Mapping ──────────────────────────────────────────

def apply_lab_tone_mapping(img, scene_type="interior"):
    """
    S-curve and gamma on L channel ONLY — a/b channels untouched.
    Reduced strength to avoid desaturation of natural tones.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32) / 255.0

    s_strength = 0.42 if scene_type == "exterior" else 0.35
    x = l_float
    curved = 1.0 / (1.0 + np.exp(-((x - 0.5) * 10 * s_strength)))
    l_float = x * (1 - s_strength * 0.5) + curved * (s_strength * 0.5)

    gamma = 0.93 if scene_type == "exterior" else 0.95
    l_float = np.power(np.clip(l_float, 0, 1), 1.0 / gamma)

    l_out = np.clip(l_float * 255, 0, 255).astype(np.uint8)
    lab_out = cv2.merge([l_out, a, b])
    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    print(f"  [ToneMap] LAB-only — S-curve: {s_strength}, Gamma: {gamma} (scene: {scene_type})")
    return result


# ─── Floor/Surface Tone Preservation ────────────────────────────────

def capture_surface_reference(img):
    """Capture LAB color signature of floor region BEFORE any processing."""
    h, w = img.shape[:2]
    floor_region = img[int(h * 0.6):, :]
    lab = cv2.cvtColor(floor_region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    ref = {
        "a_mean": np.mean(a.astype(np.float32)),
        "b_mean": np.mean(b.astype(np.float32)),
        "a_std": np.std(a.astype(np.float32)),
        "b_std": np.std(b.astype(np.float32)),
    }
    print(f"  [FloorRef] Captured — a: {ref['a_mean']:.1f}, b: {ref['b_mean']:.1f}")
    return ref


def restore_surface_tones(img, reference):
    """Correct any color drift on floor surfaces after processing."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    floor_a = a[int(h * 0.6):, :].astype(np.float32)
    floor_b = b[int(h * 0.6):, :].astype(np.float32)
    curr_a_mean = np.mean(floor_a)
    curr_b_mean = np.mean(floor_b)

    a_drift = curr_a_mean - reference["a_mean"]
    b_drift = curr_b_mean - reference["b_mean"]

    print(f"  [FloorRestore] Drift — a: {a_drift:+.1f}, b: {b_drift:+.1f}")

    if abs(a_drift) < 0.8 and abs(b_drift) < 0.8:
        print(f"  [FloorRestore] Within tolerance — no correction needed")
        return img

    a_float = a.astype(np.float32)
    b_float = b.astype(np.float32)

    gradient = np.zeros((h, w), dtype=np.float32)
    fade_start = int(h * 0.30)
    for y in range(fade_start, h):
        gradient[y, :] = (y - fade_start) / (h - fade_start)

    correction_strength = 0.90
    a_float -= a_drift * correction_strength * gradient
    b_float -= b_drift * correction_strength * gradient

    a_out = np.clip(a_float, 0, 255).astype(np.uint8)
    b_out = np.clip(b_float, 0, 255).astype(np.uint8)

    lab_out = cv2.merge([l, a_out, b_out])
    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    print(f"  [FloorRestore] Corrected a by {-a_drift * correction_strength:+.1f}, "
          f"b by {-b_drift * correction_strength:+.1f}")
    return result


# ─── Edge-Aware Contrast ────────────────────────────────────────────

def apply_edge_aware_contrast(img, scene_type="interior"):
    """Local contrast on L-channel only — zero color interference."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_float = l_channel.astype(np.float32) / 255.0

    radius = 16 if scene_type == "interior" else 20

    base = cv2.bilateralFilter(l_float, d=radius, sigmaColor=0.1, sigmaSpace=radius)
    detail = l_float - base

    detail_boost = 1.4 if scene_type == "interior" else 1.3

    shadow_mask = np.clip(1.0 - base, 0, 1) ** 0.5
    highlight_mask = np.clip(base - 0.7, 0, 0.3) / 0.3

    boosted_detail = detail * (detail_boost + shadow_mask * 0.2 - highlight_mask * 0.2)

    enhanced = base + boosted_detail
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

    lab_out = cv2.merge([enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


# ─── Adaptive Denoising ─────────────────────────────────────────────

def estimate_noise_level(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    crop = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    sigma = np.sqrt(np.maximum(cv2.Laplacian(crop, cv2.CV_64F).var() * 0.5, 0))
    return sigma


def apply_adaptive_denoising(img):
    noise_level = estimate_noise_level(img)
    print(f"  [Denoise] Estimated noise level: {noise_level:.1f}")

    if noise_level < 8:
        print(f"  [Denoise] Clean image — skipping")
        return img
    elif noise_level < 15:
        h_lum, h_color = 2, 2
    elif noise_level < 30:
        h_lum, h_color = 4, 3
    else:
        h_lum, h_color = 7, 5

    print(f"  [Denoise] Applying (h_lum={h_lum}, h_color={h_color})")
    result = cv2.fastNlMeansDenoisingColored(img, None, h_lum, h_color, 7, 21)
    return result


# ─── Auto White Balance (gentle warm-clamp) ─────────────────────────

def apply_auto_white_balance(img, scene_type="interior"):
    """
    Gentle WB: allows natural wood/material warmth through while
    preventing artificial warm casts. scale_r capped at 1.03,
    overall strength reduced to 0.12.
    """
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)

    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    scale_b = avg_all / (avg_b + 1e-6)
    scale_g = avg_all / (avg_g + 1e-6)
    scale_r = avg_all / (avg_r + 1e-6)

    max_correction = 1.12
    min_correction = 1.0 / max_correction

    scale_b = np.clip(scale_b, min_correction, max_correction)
    scale_g = np.clip(scale_g, min_correction, max_correction)
    scale_r = np.clip(scale_r, min_correction, max_correction)

    strength = 0.12 if scene_type == "interior" else 0.20

    scale_b = 1.0 + (scale_b - 1.0) * strength
    scale_g = 1.0 + (scale_g - 1.0) * strength
    scale_r = 1.0 + (scale_r - 1.0) * strength

    # SOFT WARM CLAMP: allow slight natural warmth (up to 1.03)
    if scene_type == "interior":
        scale_r = min(scale_r, 1.03)
        scale_b = max(scale_b, 0.98)

    corrected = cv2.merge([
        np.clip(b * scale_b, 0, 255),
        np.clip(g * scale_g, 0, 255),
        np.clip(r * scale_r, 0, 255)
    ]).astype(np.uint8)

    print(f"  [WB] Corrections — B: {scale_b:.3f}, G: {scale_g:.3f}, R: {scale_r:.3f} "
          f"(scene: {scene_type}, strength: {strength})")
    return corrected


# ─── Color Correction (material-aware) ──────────────────────────────

def apply_color_correction_pro(img, scene_type="interior"):
    """
    Selective neutral push with higher threshold (12.0).
    Wood floors and warm cabinets are PRESERVED, not pushed toward gray.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    a_float = a.astype(np.float32)
    b_float = b.astype(np.float32)

    chroma = np.sqrt((a_float - 128) ** 2 + (b_float - 128) ** 2)

    neutral_threshold = 12.0 if scene_type == "interior" else 8.0
    cast_mask = np.clip((chroma - neutral_threshold) / 10.0, 0, 1)

    neutral_strength = 0.03 if scene_type == "interior" else 0.05
    a_corrected = a_float + (128.0 - a_float) * neutral_strength * cast_mask
    b_corrected = b_float + (128.0 - b_float) * neutral_strength * cast_mask

    max_chroma = np.percentile(chroma, 95) + 1e-6
    vibrance_mask = 1.0 - np.clip(chroma / max_chroma, 0, 1)

    vibrance_amount = 0.10 if scene_type == "interior" else 0.15
    a_corrected = a_corrected + (a_corrected - 128) * vibrance_mask * vibrance_amount
    b_corrected = b_corrected + (b_corrected - 128) * vibrance_mask * vibrance_amount

    a_out = np.clip(a_corrected, 0, 255).astype(np.uint8)
    b_out = np.clip(b_corrected, 0, 255).astype(np.uint8)

    neutral_pct = np.mean(cast_mask < 0.1) * 100
    print(f"  [Color] Neutral push: {neutral_strength} | {neutral_pct:.0f}% pixels preserved (threshold: {neutral_threshold})")

    lab_out = cv2.merge([l, a_out, b_out])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


# ─── Shadow/Highlight Recovery ───────────────────────────────────────

def apply_shadow_highlight_recovery(img, scene_type="interior"):
    """L-channel only — reduced strength to avoid washed-out look."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32) / 255.0

    shadow_threshold = 0.45 if scene_type == "interior" else 0.30
    shadow_strength = 0.35 if scene_type == "interior" else 0.15

    shadow_mask = np.clip((shadow_threshold - l_float) / shadow_threshold, 0, 1)
    shadow_lift = shadow_mask ** 2 * shadow_strength
    l_float = l_float + shadow_lift

    highlight_threshold = 0.92
    highlight_mask = np.clip((l_float - highlight_threshold) / (1.0 - highlight_threshold), 0, 1)
    l_float = l_float - highlight_mask * 0.06

    l_out = np.clip(l_float * 255, 0, 255).astype(np.uint8)
    lab_out = cv2.merge([l_out, a, b])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


# ─── Brightness Auto-Fix ─────────────────────────────────────────────

def apply_final_brightness_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    print(f"  [Brightness] Mean: {mean_brightness:.0f}")

    result = img.copy()

    if mean_brightness < 90:
        boost = min((100 - mean_brightness) / 100 * 0.15, 0.12)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = np.clip(l.astype(np.float32) * (1 + boost) + 5, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        print(f"  [Brightness] Boosted by {boost:.1%}")
    elif mean_brightness > 190:
        reduce = min((mean_brightness - 180) / 100 * 0.1, 0.08)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = np.clip(l.astype(np.float32) * (1 - reduce), 0, 255).astype(np.uint8)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        print(f"  [Brightness] Reduced by {reduce:.1%}")

    return result


# ─── LAB-Only Sharpening ─────────────────────────────────────────────

def apply_pro_sharpening(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_float = l_channel.astype(np.float32)

    blur_fine = cv2.GaussianBlur(l_float, (3, 3), 0.8)
    detail_fine = l_float - blur_fine

    blur_med = cv2.GaussianBlur(l_float, (5, 5), 1.5)
    detail_med = l_float - blur_med

    max_detail = 12.0
    detail_fine = np.clip(detail_fine, -max_detail, max_detail)
    detail_med = np.clip(detail_med, -max_detail, max_detail)

    sharpened = l_float + detail_fine * 1.2 + detail_med * 0.4
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    lab_sharpened = cv2.merge([sharpened, a_channel, b_channel])
    return cv2.cvtColor(lab_sharpened, cv2.COLOR_LAB2BGR)


# ─── Perspective Correction ──────────────────────────────────────────

def apply_perspective_correction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=img.shape[0] // 4, maxLineGap=10)

    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < abs(y2 - y1):
            angle = np.degrees(np.arctan2(x2 - x1, y2 - y1))
            if abs(angle) < 5:
                angles.append(angle)

    if len(angles) < 3:
        return img

    median_angle = np.median(angles)

    if abs(median_angle) < 0.3 or abs(median_angle) > 5:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    result = cv2.warpAffine(img, M, (w, h),
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REFLECT101)
    print(f"  [Perspective] Corrected by {median_angle:.2f}°")
    return result


# ─── Main Pipeline ───────────────────────────────────────────────────

def process_hdr(images, style="natural"):
    print(f"[HDR Engine v5.2] Processing {len(images)} frames, style: {style}")
    t0 = time.time()

    scene_type = detect_scene_type(images)

    print("[Step 1/9] Aligning frames...")
    aligned = align_images_ecc(images)

    print("[Step 2/9] Ghost removal...")
    deghosted = remove_ghosts(aligned)

    # Capture floor reference from MIDDLE BRACKET (before fusion)
    print("[Step 3/9] Capturing surface tone reference from source bracket...")
    floor_ref = capture_surface_reference(images[len(images) // 2])

    print("[Step 4/9] Exposure fusion...")
    merged = exposure_fusion(deghosted, scene_type)

    print("[Step 5/9] White balance (gentle warm-clamp)...")
    merged = apply_auto_white_balance(merged, scene_type)

    print("[Step 6/9] LAB tone mapping...")
    merged = apply_lab_tone_mapping(merged, scene_type)

    print("[Step 7/9] Edge-aware contrast...")
    merged = apply_edge_aware_contrast(merged, scene_type)

    print("[Step 8/9] Color correction (material-aware)...")
    merged = apply_color_correction_pro(merged, scene_type)

    print("[Step 9/9] Shadow/highlight recovery...")
    merged = apply_shadow_highlight_recovery(merged, scene_type)

    merged = apply_adaptive_denoising(merged)
    merged = apply_final_brightness_contrast(merged)

    print("[Floor Restore] Checking for color drift vs source...")
    merged = restore_surface_tones(merged, floor_ref)

    merged = apply_perspective_correction(merged)
    merged = apply_pro_sharpening(merged)

    elapsed = time.time() - t0
    print(f"[HDR Engine v5.2] Done in {elapsed:.1f}s (scene: {scene_type})")
    return merged


# ─── Job Handler ─────────────────────────────────────────────────────

def process_job(payload):
    job_id = payload.get("jobId") or payload.get("job_id") or "unknown"
    input_urls = payload.get("inputUrls") or payload.get("input_urls") or []
    style = payload.get("style", "natural")
    webhook_url = payload.get("webhookUrl") or payload.get("webhook_url")
    webhook_secret = payload.get("webhookSecret") or payload.get("webhook_secret")

    print(f"\n{'='*60}")
    print(f"[Job {job_id}] Starting HDR processing (v5.2)")
    print(f"[Job {job_id}] Inputs: {len(input_urls)}, Style: {style}")
    print(f"[Job {job_id}] Webhook: {'yes' if webhook_url else 'no'}")
    print(f"{'='*60}")

    try:
        images = []
        for i, url in enumerate(input_urls):
            print(f"[Job {job_id}] Downloading image {i+1}/{len(input_urls)}...")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image {i+1}")
            print(f"[Job {job_id}] Image {i+1}: {img.shape[1]}x{img.shape[0]}")
            images.append(img)

        if len(images) < 2:
            raise ValueError("Need at least 2 images for HDR merge")

        result = process_hdr(images, style)

        # Full quality encode for final result
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        success, encoded = cv2.imencode(".jpg", result, encode_params)
        if not success:
            raise ValueError("Failed to encode result image")

        result_bytes = encoded.tobytes()
        print(f"[Job {job_id}] Result (full quality): {len(result_bytes)/1024:.0f}KB")

        if webhook_url:
            import base64

            # Re-compress for webhook transfer to prevent timeout
            # The hdr-webhook edge function re-enhances with AI anyway
            webhook_quality = 85
            _, webhook_encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, webhook_quality])
            webhook_bytes = webhook_encoded.tobytes()
            print(f"[Job {job_id}] Webhook payload: {len(webhook_bytes)/1024:.0f}KB "
                  f"(compressed from {len(result_bytes)/1024:.0f}KB)")

            result_b64 = base64.b64encode(webhook_bytes).decode("utf-8")

            callback_payload = {
                "jobId": job_id,
                "job_id": job_id,
                "status": "completed",
                "result": result_b64,
            }

            headers = {"Content-Type": "application/json"}
            if webhook_secret:
                headers["X-Webhook-Secret"] = webhook_secret
                headers["x-webhook-secret"] = webhook_secret

            print(f"[Job {job_id}] Sending webhook to {webhook_url}...")
            resp = requests.post(
                webhook_url,
                json=callback_payload,
                headers=headers,
                timeout=120
            )
            print(f"[Job {job_id}] Webhook response: {resp.status_code}")

        print(f"[Job {job_id}] ✅ Complete!")
        return {"status": "completed", "jobId": job_id}

    except Exception as e:
        error_msg = str(e)
        print(f"[Job {job_id}] ❌ Error: {error_msg}")
        traceback.print_exc()

        if webhook_url:
            try:
                error_payload = {
                    "jobId": job_id,
                    "job_id": job_id,
                    "status": "failed",
                    "error": error_msg,
                }
                headers = {"Content-Type": "application/json"}
                if webhook_secret:
                    headers["X-Webhook-Secret"] = webhook_secret
                    headers["x-webhook-secret"] = webhook_secret

                requests.post(webhook_url, json=error_payload,
                              headers=headers, timeout=30)
            except Exception as we:
                print(f"[Job {job_id}] Webhook error notification failed: {we}")

        raise


# ─── Worker Entry ─────────────────────────────────────────────────────

if __name__ == "__main__":
    redis_conn = Redis.from_url(REDIS_URL)
    queue = Queue("hdr", connection=redis_conn)
    worker = Worker([queue], connection=redis_conn)
    print("[Worker] Starting HDR worker v5.2 — balanced warmth + webhook compression")
    worker.work(with_scheduler=False)
