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
# IO
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


# -----------------------------
# Brightness / ordering
# -----------------------------
def luminance_L01(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].astype(np.float32) / 255.0


def sort_by_brightness(images: list) -> list:
    scores = [float(np.mean(luminance_L01(im))) for im in images]
    idx = np.argsort(scores)  # darkest -> brightest
    return [images[i] for i in idx]


def order_images(images: list, order: str) -> list:
    # We always sort by brightness because user-upload order is unreliable.
    # order flag kept for UI compatibility.
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

            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 250, 1e-6)

            small_ref = cv2.resize(ref_gray, None, fx=0.5, fy=0.5)
            small_gray = cv2.resize(gray, None, fx=0.5, fy=0.5)

            _, warp = cv2.findTransformECC(
                small_ref, small_gray, warp, cv2.MOTION_EUCLIDEAN, criteria
            )

            warp[0, 2] *= 2
            warp[1, 2] *= 2

            h, w = ref.shape[:2]
            out = cv2.warpAffine(
                img, warp, (w, h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
            aligned.append(out)
        except Exception as e:
            print(f"[Worker] Alignment failed on image {i}: {e}")
            aligned.append(img)

    return aligned


# -----------------------------
# Ghost suppression (simple + safe)
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
        _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8)

        a = (mask.astype(np.float32) / 255.0)[:, :, None]
        blended = img.astype(np.float32) * (1 - a) + ref.astype(np.float32) * a
        out.append(np.clip(blended, 0, 255).astype(np.uint8))

    return out


# -----------------------------
# Fusion base (avoid crunchy HDR)
# -----------------------------
def exposure_fusion(images: list) -> np.ndarray:
    merge = cv2.createMergeMertens(
        contrast_weight=0.95,
        saturation_weight=0.55,
        exposure_weight=1.10,
    )
    fusion = merge.process(images)
    fusion = np.clip(fusion * 255.0, 0, 255).astype(np.uint8)
    return fusion


# -----------------------------
# Window Pull (AGGRESSIVE but CONTROLLED)
# Uses DARK frame ONLY where highlights are blown.
# Includes “mask guard” to stop the whole image turning black.
# -----------------------------
def window_pull(fused: np.ndarray, dark: np.ndarray, strength: float) -> np.ndarray:
    # Strength: 0.0..1.0
    L = luminance_L01(fused)

    # highlight threshold (percentile based)
    t = np.percentile(L, 96.0 if strength < 0.7 else 94.0)

    mask = (L >= t).astype(np.uint8) * 255

    # remove speckles & keep smooth areas like windows
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((19, 19), np.uint8), iterations=2)

    # feather heavily to avoid halos
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=26)
    mask_f = np.clip(mask_f, 0.0, 1.0)

    # GUARD: if mask covers too much, tighten threshold until it’s sane
    # (this prevents the “black everywhere” disaster you showed)
    for _ in range(8):
        if float(np.mean(mask_f)) <= 0.14:
            break
        t = min(t + 0.02, 0.995)
        mask = (L >= t).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((17, 17), np.uint8), iterations=1)
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=26)
        mask_f = np.clip(mask_f, 0.0, 1.0)

    # blend
    m = mask_f * (0.55 + 0.40 * strength)  # 0.55..0.95
    m3 = np.dstack([m, m, m])

    out = fused.astype(np.float32) * (1.0 - m3) + dark.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------------
# Tone / brightness normalization (fix “too dark/grey”)
# -----------------------------
def normalize_exposure(img: np.ndarray, target_mid: float = 0.58) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0

    mid = float(np.percentile(L, 50))
    hi = float(np.percentile(L, 99))

    mid = max(mid, 1e-4)
    gain = target_mid / mid
    gain = np.clip(gain, 0.85, 2.50)

    L = np.clip(L * gain, 0, 1)

    # compress highlights gently so it looks “pro” not blown
    # keeps ceiling lights from clipping hard
    if hi > 0.92:
        L = 1.0 - np.power(1.0 - L, 1.12)

    lab[:, :, 0] = np.clip(L * 255.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def filmic_contrast(img: np.ndarray, strength: float = 0.35) -> np.ndarray:
    # smooth “S-curve” on L channel (no crunchy HDR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0

    # sigmoid
    a = 6.0 * strength + 1.0  # 1..3-ish
    L2 = 1.0 / (1.0 + np.exp(-a * (L - 0.5)))
    # mix
    L = (1 - strength) * L + strength * L2

    lab[:, :, 0] = np.clip(L * 255.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# -----------------------------
# White balance (safe)
# -----------------------------
def balanced_awb(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    a_mean = float(np.mean(A))
    b_mean = float(np.mean(B))

    # gentle correction only
    A = A - (a_mean - 128) * 0.10
    B = B - (b_mean - 128) * 0.10

    lab = cv2.merge([L, np.clip(A, 0, 255), np.clip(B, 0, 255)])
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# -----------------------------
# Local contrast (“clarity”) WITHOUT halos
# -----------------------------
def clarity(img: np.ndarray, amount: float = 0.40) -> np.ndarray:
    # edge-preserving base
    base = cv2.bilateralFilter(img, d=0, sigmaColor=40, sigmaSpace=18)
    detail = img.astype(np.float32) - base.astype(np.float32)
    out = img.astype(np.float32) + detail * (amount * 1.8)
    return np.clip(out, 0, 255).astype(np.uint8)


def mild_saturation(img: np.ndarray, sat_boost: float = 0.08) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1.0 + sat_boost)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
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
    style = payload.get("style", "natural")           # natural | window_soft | window_strong
    order = payload.get("order", "random")           # random | dark | normal | bright
    webhook_url = payload.get("webhook_url")
    webhook_secret = payload.get("webhook_secret")

    result_key = f"hdr_result:{job_id}"
    redis_conn.hset(result_key, mapping={"status": "processing"})
    redis_conn.expire(result_key, 3600)

    print(f"[Worker] Processing job {job_id} ({len(input_urls)} imgs) style={style} order={order}")
    start = time.time()

    try:
        # Download
        images = [download_image(u) for u in input_urls]
        if len(images) < 2:
            raise ValueError("Need at least 2 images")

        # Normalize sizes
        ref = images[0]
        images = [safe_resize_like(im, ref) for im in images]

        # Brightness order
        images = order_images(images, order)
        dark = images[0]
        mid = images[len(images)//2]
        bright = images[-1]

        # Align + ghost suppression
        images = align_images(images)
        images = remove_ghosts(images)

        # Fusion base
        fused = exposure_fusion(images)

        # Window pull (only if requested)
        if style == "window_strong":
            fused = window_pull(fused, dark, strength=1.0)
        elif style == "window_soft":
            fused = window_pull(fused, dark, strength=0.55)

        # Normalize (prevents grey/dull output)
        fused = normalize_exposure(fused, target_mid=0.60)

        # Filmic contrast (clean, no crunchy HDR)
        fused = filmic_contrast(fused, strength=0.33)

        # White balance
        fused = balanced_awb(fused)

        # Local contrast (clarity)
        fused = clarity(fused, amount=0.40)

        # Mild color (MLS safe)
        fused = mild_saturation(fused, sat_boost=0.08)

        # Clean finishing
        fused = denoise(fused)
        fused = sharpen_soft(fused)

        # Output to Redis as a data URL (so /result works)
        result_bytes = encode_jpeg(fused, FINAL_JPEG_QUALITY)
        output_url = to_data_url(result_bytes)

        redis_conn.hset(result_key, mapping={"status": "completed", "output_url": output_url})
        redis_conn.expire(result_key, 3600)

        # Webhook send
        if webhook_url:
            webhook_bytes = encode_jpeg(fused, WEBHOOK_JPEG_QUALITY)
            webhook_payload = {
                "jobId": job_id,
                "status": "completed",
                "result": base64.b64encode(webhook_bytes).decode("utf-8"),
            }
            try:
                send_webhook(webhook_url, webhook_secret, webhook_payload)
                print(f"[Worker] Webhook sent: {webhook_url}")
            except Exception as e:
                print(f"[Worker] Webhook failed: {e}")

        print(f"[Worker] Done {job_id} in {time.time()-start:.1f}s")

    except Exception as e:
        err = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Worker] FAILED {job_id}:\n{err}")

        redis_conn.hset(result_key, mapping={"status": "failed", "error": str(e)})
        redis_conn.expire(result_key, 3600)

        if webhook_url:
            try:
                send_webhook(webhook_url, webhook_secret, {"jobId": job_id, "status": "failed", "error": str(e)})
            except Exception:
                pass
