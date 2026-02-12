import os
import io
import cv2
import json
import time
import redis
import base64
import requests
import traceback
import numpy as np
from PIL import Image

FINAL_JPEG_QUALITY = 97
WEBHOOK_JPEG_QUALITY = 85

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_conn = redis.from_url(REDIS_URL)


def download_image(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from URL")
    return img


def detect_scene(images: list) -> str:
    img = images[len(images) // 2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mean_sat = float(np.mean(s))
    mean_val = float(np.mean(v))
    bright_pct = float(np.sum(v > 200)) / v.size
    if bright_pct > 0.25 and mean_sat > 60:
        return "exterior"
    return "interior"


def correct_lens(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    fx = fy = max(w, h)
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([-0.08, 0.02, 0, 0, 0], dtype=np.float64)
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0.3, (w, h))
    corrected = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_mtx)
    x, y, rw, rh = roi
    if rw > w * 0.8 and rh > h * 0.8:
        corrected = corrected[y:y+rh, x:x+rw]
        corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return corrected


def remove_vignette(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r_max = np.sqrt(cx**2 + cy**2)
    r_norm = r / r_max
    gain = 1.0 + 0.35 * (r_norm ** 2.5)
    gain = np.clip(gain, 1.0, 1.6)
    gain_3ch = np.dstack([gain, gain, gain]).astype(np.float32)
    result = np.clip(img.astype(np.float32) * gain_3ch, 0, 255).astype(np.uint8)
    return result


def correct_perspective(img: np.ndarray, scene: str) -> np.ndarray:
    if scene != "interior":
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) < 5:
        return img
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15 or abs(abs(angle) - 90) < 15:
            angles.append(angle)
    if not angles:
        return img
    near_zero = [a for a in angles if abs(a) < 15]
    if near_zero:
        median_angle = float(np.median(near_zero))
        if abs(median_angle) > 0.3 and abs(median_angle) < 5.0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    return img


def align_images(images: list) -> list:
    if len(images) < 2:
        return images
    ref = images[len(images) // 2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    aligned = []
    for i, img in enumerate(images):
        if i == len(images) // 2:
            aligned.append(ref)
            continue
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.shape != ref_gray.shape:
                gray = cv2.resize(gray, (ref_gray.shape[1], ref_gray.shape[0]))
                img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
            small_ref = cv2.resize(ref_gray, None, fx=0.5, fy=0.5)
            small_gray = cv2.resize(gray, None, fx=0.5, fy=0.5)
            _, warp_matrix = cv2.findTransformECC(small_ref, small_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
            warp_matrix[0, 2] *= 2
            warp_matrix[1, 2] *= 2
            h, w = ref.shape[:2]
            aligned_img = cv2.warpAffine(img, warp_matrix, (w, h), flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
            aligned.append(aligned_img)
        except Exception as e:
            print(f"[Worker] Alignment failed for image {i}, using original: {e}")
            if img.shape != ref.shape:
                img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
            aligned.append(img)
    return aligned


def remove_ghosts(images: list) -> list:
    if len(images) < 3:
        return images
    ref = images[len(images) // 2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    result = []
    for i, img in enumerate(images):
        if i == len(images) // 2:
            result.append(img)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = cv2.absdiff(gray, ref_gray)
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        alpha = (mask / 255.0)[:, :, np.newaxis]
        blended = (img.astype(np.float32) * (1 - alpha) + ref.astype(np.float32) * alpha)
        result.append(np.clip(blended, 0, 255).astype(np.uint8))
    return result


def exposure_fusion(images: list) -> np.ndarray:
    merge = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=0.8,
        exposure_weight=0.6,
    )
    fusion = merge.process(images)
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion


def recover_highlights_shadows(fused: np.ndarray, dark: np.ndarray) -> np.ndarray:
    lab_fused = cv2.cvtColor(fused, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_dark = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_fused = lab_fused[:, :, 0]
    L_dark = lab_dark[:, :, 0]
    highlight_mask = np.clip((L_fused - 200) / 55.0, 0, 1)
    lab_fused[:, :, 0] = L_fused * (1 - highlight_mask * 0.4) + L_dark * (highlight_mask * 0.4)
    shadow_mask = np.clip((60 - lab_fused[:, :, 0]) / 60.0, 0, 1)
    lab_fused[:, :, 0] += shadow_mask * 15
    lab_fused[:, :, 0] = np.clip(lab_fused[:, :, 0], 0, 255)
    return cv2.cvtColor(lab_fused.astype(np.uint8), cv2.COLOR_LAB2BGR)


def tone_map_lab(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0
    L = np.power(L, 0.91)
    mid = 0.5
    strength = 0.50
    L = mid + (L - mid) * (1 + strength * (0.5 - np.abs(L - mid)))
    lab[:, :, 0] = np.clip(L * 255, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def balanced_awb(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    a_mean = float(np.mean(A))
    b_mean = float(np.mean(B))
    strength = 0.12
    A = A - (a_mean - 128) * strength
    B = B - (b_mean - 128) * strength
    B = np.clip(B, 128 - 8, 128 + 3)
    lab = cv2.merge([L, np.clip(A, 0, 255), np.clip(B, 0, 255)])
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def preserve_surface_tones(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    lab_orig = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_orig = lab_orig[:, :, 0]
    neutral_mask = np.ones_like(L_orig)
    a_orig = lab_orig[:, :, 1]
    b_orig = lab_orig[:, :, 2]
    chroma = np.sqrt((a_orig - 128)**2 + (b_orig - 128)**2)
    neutral_mask = np.clip(1.0 - chroma / 30.0, 0, 1)
    neutral_mask = cv2.GaussianBlur(neutral_mask, (31, 31), 0)
    blend_strength = 0.3
    alpha = neutral_mask * blend_strength
    alpha_3ch = np.dstack([alpha, alpha, alpha])
    lab_result = lab_proc * (1 - alpha_3ch) + lab_orig * alpha_3ch
    lab_result = np.clip(lab_result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)


def apply_clahe(img: np.ndarray, scene: str) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clip_limit = 1.5 if scene == "interior" else 1.8
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    L = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)


def dehaze(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    blur = cv2.GaussianBlur(L, (0, 0), sigmaX=40)
    detail = L - blur
    L = L + detail * 0.2
    lab[:, :, 0] = np.clip(L, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def adaptive_denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 7, 21)


def sharpen(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    blur1 = cv2.GaussianBlur(L, (0, 0), sigmaX=1.0)
    sharp_fine = cv2.addWeighted(L, 1.3, blur1, -0.3, 0)
    blur2 = cv2.GaussianBlur(sharp_fine, (0, 0), sigmaX=2.0)
    sharp_mid = cv2.addWeighted(sharp_fine, 1.15, blur2, -0.15, 0)
    return cv2.cvtColor(cv2.merge([sharp_mid, A, B]), cv2.COLOR_LAB2BGR)


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def process_job(payload: dict):
    job_id = payload["job_id"]
    input_urls = payload["input_urls"]
    style = payload.get("style", "natural")
    webhook_url = payload.get("webhook_url")
    webhook_secret = payload.get("webhook_secret")

    result_key = f"hdr_result:{job_id}"
    redis_conn.hset(result_key, mapping={"status": "processing"})
    redis_conn.expire(result_key, 3600)

    print(f"[Worker] Processing job {job_id} with {len(input_urls)} images, style={style}")
    start_time = time.time()

    try:
        # Step 1: Download
        print(f"[Worker] Step 1/16: Downloading {len(input_urls)} images...")
        images = [download_image(url) for url in input_urls]
        print(f"[Worker] Downloaded. Sizes: {[img.shape for img in images]}")

        # Step 2: Scene detection
        print("[Worker] Step 2/16: Scene detection...")
        scene = detect_scene(images)
        print(f"[Worker] Scene: {scene}")

        # Step 3: Lens correction
        print("[Worker] Step 3/16: Lens correction...")
        images = [correct_lens(img) for img in images]

        # Step 4: Vignette removal
        print("[Worker] Step 4/16: Vignette removal...")
        images = [remove_vignette(img) for img in images]

        # Step 5: Perspective correction
        print("[Worker] Step 5/16: Perspective correction...")
        images = [correct_perspective(img, scene) for img in images]

        # Step 6: Sub-pixel alignment
        print("[Worker] Step 6/16: Sub-pixel ECC alignment...")
        images = align_images(images)

        # Step 7: Ghost removal
        print("[Worker] Step 7/16: Ghost removal...")
        images = remove_ghosts(images)

        # Step 8: Exposure fusion
        print("[Worker] Step 8/16: Exposure fusion (Mertens)...")
        pre_fusion_lab = cv2.cvtColor(images[len(images)//2], cv2.COLOR_BGR2LAB).astype(np.float32)
        fused = exposure_fusion(images)

        # Step 9: Highlight/shadow recovery
        print("[Worker] Step 9/16: Highlight & shadow recovery...")
        dark_img = images[0]
        fused = recover_highlights_shadows(fused, dark_img)

        # Step 10: LAB tone mapping
        print("[Worker] Step 10/16: LAB tone mapping...")
        fused = tone_map_lab(fused)

        # Step 11: Balanced AWB
        print("[Worker] Step 11/16: Balanced AWB...")
        fused = balanced_awb(fused)

        # Step 12: Surface tone preservation
        print("[Worker] Step 12/16: Surface tone preservation...")
        fused = preserve_surface_tones(images[len(images)//2], fused)

        # Step 13: CLAHE contrast
        print("[Worker] Step 13/16: Edge-aware contrast (CLAHE)...")
        fused = apply_clahe(fused, scene)

        # Step 14: Dehaze
        print("[Worker] Step 14/16: Dehazing...")
        fused = dehaze(fused)

        # Step 15: Adaptive denoise
        print("[Worker] Step 15/16: Adaptive denoising...")
        fused = adaptive_denoise(fused)

        # Step 16: Multi-scale sharpening
        print("[Worker] Step 16/16: Multi-scale sharpening...")
        fused = sharpen(fused)

        elapsed = time.time() - start_time
        print(f"[Worker] Processing complete in {elapsed:.1f}s. Output shape: {fused.shape}")

        # Encode result
        result_bytes = encode_jpeg(fused, FINAL_JPEG_QUALITY)
        print(f"[Worker] Final JPEG size: {len(result_bytes) / 1024:.0f} KB")

        # Send webhook
        if webhook_url:
            print(f"[Worker] Sending webhook to {webhook_url}")
            webhook_bytes = encode_jpeg(fused, WEBHOOK_JPEG_QUALITY)
            b64 = base64.b64encode(webhook_bytes).decode("utf-8")
            webhook_payload = {
                "jobId": job_id,
                "status": "completed",
                "result": b64,
            }
            headers = {"Content-Type": "application/json"}
            if webhook_secret:
                headers["x-webhook-secret"] = webhook_secret
            try:
                resp = requests.post(webhook_url, json=webhook_payload, headers=headers, timeout=120)
                print(f"[Worker] Webhook response: {resp.status_code}")
            except Exception as e:
                print(f"[Worker] Webhook failed: {e}")

        # Store in Redis
        redis_conn.hset(result_key, mapping={"status": "completed", "output_url": "webhook_delivered"})
        redis_conn.expire(result_key, 3600)
        print(f"[Worker] Job {job_id} done.")

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Worker] Job {job_id} FAILED after {elapsed:.1f}s: {error_msg}")
        redis_conn.hset(result_key, mapping={"status": "failed", "error": str(e)})
        redis_conn.expire(result_key, 3600)

        if webhook_url:
            try:
                headers = {"Content-Type": "application/json"}
                if webhook_secret:
                    headers["x-webhook-secret"] = webhook_secret
                requests.post(webhook_url, json={"jobId": job_id, "status": "failed", "error": str(e)}, headers=headers, timeout=30)
            except Exception:
                pass
