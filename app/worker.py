import os
import time
import redis
import base64
import requests
import traceback
import numpy as np
import cv2

FINAL_JPEG_QUALITY = 97
WEBHOOK_JPEG_QUALITY = 92

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
redis_conn = redis.from_url(REDIS_URL)


# -----------------------------
# Helpers
# -----------------------------
def download_image(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from URL")
    return img


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()


def to_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def safe_resize_like(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if img.shape[:2] == ref.shape[:2]:
        return img
    return cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)


def luminance_L(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].astype(np.float32) / 255.0


# -----------------------------
# Scene detection
# -----------------------------
def detect_scene(images: list) -> str:
    img = images[len(images) // 2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mean_sat = float(np.mean(s))
    bright_pct = float(np.sum(v > 200)) / v.size
    if bright_pct > 0.25 and mean_sat > 60:
        return "exterior"
    return "interior"


# -----------------------------
# Ordering: random/dark/normal/bright
# Always sorts to be safe because user upload order is inconsistent.
# -----------------------------
def sort_by_brightness(images: list) -> list:
    scores = []
    for im in images:
        scores.append(float(np.mean(luminance_L(im))))
    idx = np.argsort(scores)
    return [images[i] for i in idx]


def order_images(images: list, order: str) -> list:
    # For all orders, we sort by brightness for reliability.
    # 'order' is kept for future logic if you want strict mode.
    return sort_by_brightness(images)


# -----------------------------
# Alignment
# -----------------------------
def align_images(images: list) -> list:
    if len(images) < 2:
        return images

    ref = images[len(images) // 2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    aligned = []
    for i, img in enumerate(images):
        img = safe_resize_like(img, ref)

        if i == len(images) // 2:
            aligned.append(ref)
            continue

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

            small_ref = cv2.resize(ref_gray, None, fx=0.5, fy=0.5)
            small_gray = cv2.resize(gray, None, fx=0.5, fy=0.5)

            _, warp_matrix = cv2.findTransformECC(
                small_ref, small_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
            )
            warp_matrix[0, 2] *= 2
            warp_matrix[1, 2] *= 2

            h, w = ref.shape[:2]
            aligned_img = cv2.warpAffine(
                img, warp_matrix, (w, h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
            aligned.append(aligned_img)

        except Exception as e:
            print(f"[Worker] Alignment failed for image {i}: {e}")
            aligned.append(img)

    return aligned


# -----------------------------
# Ghost removal
# -----------------------------
def remove_ghosts(images: list) -> list:
    if len(images) < 3:
        return images

    ref = images[len(images) // 2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

    out = []
    for i, img in enumerate(images):
        img = safe_resize_like(img, ref)
        if i == len(images) // 2:
            out.append(img)
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
        out.append(np.clip(blended, 0, 255).astype(np.uint8))

    return out


# -----------------------------
# Exposure fusion (base)
# -----------------------------
def exposure_fusion(images: list) -> np.ndarray:
    merge = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=0.65,   # reduces HDR “crunch”
        exposure_weight=0.90      # improves balanced exposure
    )
    fusion = merge.process(images)
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion


# -----------------------------
# Window pull: SAFE + STRONG (halo guarded)
# -----------------------------
def window_pull_soft(fused: np.ndarray, dark: np.ndarray) -> np.ndarray:
    L = luminance_L(fused)
    t = np.percentile(L, 96)

    mask = (L >= t).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=2)

    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=18)
    mask_f = np.clip(mask_f, 0.0, 1.0)

    # Guard: if mask covers too much, tighten threshold
    if float(np.mean(mask_f)) > 0.18:
        t = np.percentile(L, 98)
        mask = (L >= t).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=18)
        mask_f = np.clip(mask_f, 0.0, 1.0)

    m = mask_f * 0.85
    m3 = np.dstack([m, m, m])

    out = fused.astype(np.float32) * (1.0 - m3) + dark.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


def window_pull_strong(fused: np.ndarray, dark: np.ndarray) -> np.ndarray:
    L = luminance_L(fused)

    t = np.percentile(L, 94)
    target_max = 0.14

    for _ in range(8):
        mask = (L >= t).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((17, 17), np.uint8), iterations=2)

        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=24)
        mask_f = np.clip(mask_f, 0.0, 1.0)

        if float(np.mean(mask_f)) <= target_max:
            break

        t = min(t + 1.5, 99.2)

    # extra feather reduces halos
    mask_f = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=12)
    mask_f = np.clip(mask_f, 0.0, 1.0)

    m = mask_f * 0.90
    m3 = np.dstack([m, m, m])

    out = fused.astype(np.float32) * (1.0 - m3) + dark.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------------
# Brightness + clean look (avoid too dark)
# -----------------------------
def auto_exposure(img: np.ndarray, target_mid: float = 0.58) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0

    med = float(np.median(L))
    med = max(med, 1e-4)

    gain = target_mid / med
    gain = np.clip(gain, 0.75, 2.60)

    L = np.clip(L * gain, 0, 1)

    # gentle shadow lift if dark overall
    if med < 0.40:
        L = np.power(L, 0.88)

    lab[:, :, 0] = L * 255.0
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def balanced_awb(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    a_mean = float(np.mean(A))
    b_mean = float(np.mean(B))

    strength = 0.10
    A = A - (a_mean - 128) * strength
    B = B - (b_mean - 128) * strength

    B = np.clip(B, 128 - 10, 128 + 6)

    lab = cv2.merge([L, np.clip(A, 0, 255), np.clip(B, 0, 255)])
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_clahe(img: np.ndarray, scene: str) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clip_limit = 1.2 if scene == "interior" else 1.4
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    L = clahe.apply(L)

    return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)


def adaptive_denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 7, 21)


def sharpen_soft(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    blur = cv2.GaussianBlur(L, (0, 0), sigmaX=1.2)
    sharp = cv2.addWeighted(L, 1.22, blur, -0.22, 0)

    return cv2.cvtColor(cv2.merge([sharp, A, B]), cv2.COLOR_LAB2BGR)


def send_webhook(webhook_url: str, webhook_secret: str | None, payload: dict):
    headers = {"Content-Type": "application/json"}
    if webhook_secret:
        headers["x-webhook-secret"] = webhook_secret
    requests.post(webhook_url, json=payload, headers=headers, timeout=120)


# -----------------------------
# Main job
# -----------------------------
def process_job(payload: dict):
    job_id = payload["job_id"]
    input_urls = payload["input_urls"]
    style = payload.get("style", "natural")
    order = payload.get("order", "random")
    webhook_url = payload.get("webhook_url")
    webhook_secret = payload.get("webhook_secret")

    result_key = f"hdr_result:{job_id}"
    redis_conn.hset(result_key, mapping={"status": "processing"})
    redis_conn.expire(result_key, 3600)

    print(f"[Worker] Processing job {job_id} ({len(input_urls)} imgs) style={style} order={order}")
    start_time = time.time()

    try:
        # 1) Download
        images = [download_image(url) for url in input_urls]
        if len(images) < 2:
            raise ValueError("Need at least 2 images")

        # 2) Normalize sizes
        ref = images[0]
        images = [safe_resize_like(im, ref) for im in images]

        # 3) Sort brightness (so we reliably pick dark/mid/bright)
        images = order_images(images, order)
        dark_img = images[0]
        mid_img = images[len(images) // 2]
        bright_img = images[-1]

        # 4) Scene detection
        scene = detect_scene(images)
        print(f"[Worker] Scene: {scene}")

        # 5) Align
        images = align_images(images)

        # 6) Ghost remove
        images = remove_ghosts(images)

        # 7) Fusion
        fused = exposure_fusion(images)

        # 8) Window pull
        if style == "window_strong":
            fused = window_pull_strong(fused, dark_img)
        elif style == "window_soft":
            fused = window_pull_soft(fused, dark_img)

        # 9) Fix overall brightness
        fused = auto_exposure(fused, target_mid=0.58 if scene == "interior" else 0.55)

        # 10) AWB
        fused = balanced_awb(fused)

        # 11) Gentle contrast
        fused = apply_clahe(fused, scene)

        # 12) Clean
        fused = adaptive_denoise(fused)
        fused = sharpen_soft(fused)

        elapsed = time.time() - start_time
        print(f"[Worker] Complete in {elapsed:.1f}s. Output shape: {fused.shape}")

        # Store output as a data URL so /result works
        result_bytes = encode_jpeg(fused, FINAL_JPEG_QUALITY)
        output_url = to_data_url(result_bytes)

        redis_conn.hset(result_key, mapping={"status": "completed", "output_url": output_url})
        redis_conn.expire(result_key, 3600)

        # Webhook
        if webhook_url:
            webhook_bytes = encode_jpeg(fused, WEBHOOK_JPEG_QUALITY)
            webhook_payload = {
                "jobId": job_id,
                "status": "completed",
                "result": base64.b64encode(webhook_bytes).decode("utf-8"),
            }
            try:
                send_webhook(webhook_url, webhook_secret, webhook_payload)
                print(f"[Worker] Webhook sent to {webhook_url}")
            except Exception as e:
                print(f"[Worker] Webhook failed: {e}")

        print(f"[Worker] Job {job_id} done.")

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Worker] Job {job_id} FAILED after {elapsed:.1f}s:\n{error_msg}")

        redis_conn.hset(result_key, mapping={"status": "failed", "error": str(e)})
        redis_conn.expire(result_key, 3600)

        if webhook_url:
            try:
                send_webhook(webhook_url, webhook_secret, {"jobId": job_id, "status": "failed", "error": str(e)})
            except Exception:
                pass
