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
# Download
# -----------------------------
def download_image(url: str) -> np.ndarray:
    resp = requests.get(str(url), timeout=120)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from URL")
    return img


# -----------------------------
# Scene detection (simple)
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
# Optional: mild lens correction (keep, but safe)
# -----------------------------
def correct_lens(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    fx = fy = max(w, h)
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # mild distortion values (your old ones were ok)
    dist_coeffs = np.array([-0.06, 0.015, 0, 0, 0], dtype=np.float64)

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0.3, (w, h))
    corrected = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_mtx)
    x, y, rw, rh = roi

    if rw > w * 0.85 and rh > h * 0.85:
        corrected = corrected[y:y + rh, x:x + rw]
        corrected = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return corrected


def remove_vignette(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = np.sqrt(cx ** 2 + cy ** 2)
    r_norm = r / r_max

    gain = 1.0 + 0.25 * (r_norm ** 2.2)  # reduced from 0.35 to avoid gray wash
    gain = np.clip(gain, 1.0, 1.45)

    gain_3 = np.dstack([gain, gain, gain]).astype(np.float32)
    out = np.clip(img.astype(np.float32) * gain_3, 0, 255).astype(np.uint8)
    return out


# -----------------------------
# Alignment (ECC)
# -----------------------------
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

            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

            small_ref = cv2.resize(ref_gray, None, fx=0.5, fy=0.5)
            small_gray = cv2.resize(gray, None, fx=0.5, fy=0.5)

            _, warp = cv2.findTransformECC(
                small_ref, small_gray, warp,
                motionType=cv2.MOTION_EUCLIDEAN,
                criteria=criteria
            )

            warp[0, 2] *= 2
            warp[1, 2] *= 2

            h, w = ref.shape[:2]
            aligned_img = cv2.warpAffine(
                img, warp, (w, h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
            aligned.append(aligned_img)

        except Exception as e:
            print(f"[Worker] Alignment failed img {i}: {e}")
            if img.shape != ref.shape:
                img = cv2.resize(img, (ref.shape[1], ref.shape[0]))
            aligned.append(img)

    return aligned


# -----------------------------
# Ghost handling (keep mild)
# -----------------------------
def remove_ghosts(images: list) -> list:
    if len(images) < 3:
        return images

    ref = images[len(images) // 2]
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)

    out = []
    for i, img in enumerate(images):
        if i == len(images) // 2:
            out.append(img)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = cv2.absdiff(gray, ref_gray)

        _, mask = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)  # slightly lower
        mask = mask.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10)

        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]

        blended = img.astype(np.float32) * (1 - alpha) + ref.astype(np.float32) * alpha
        out.append(np.clip(blended, 0, 255).astype(np.uint8))

    return out


# -----------------------------
# New HDR pipeline helpers
# -----------------------------
def to_float01(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)

def to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)

def luminance_L(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0].astype(np.float32)  # 0..255

def exposure_fusion_better(images: list) -> np.ndarray:
    merge = cv2.createMergeMertens(
        contrast_weight=0.60,
        saturation_weight=0.35,
        exposure_weight=1.00,
    )
    imgs01 = [to_float01(im) for im in images]
    fusion01 = merge.process(imgs01)
    return to_uint8(fusion01)

def window_pull_soft(fused: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """
    Soft window recovery:
    - mask from fused highlights
    - small coverage target (prevents darkening the whole image)
    """
    L = luminance_L(fused)

    # Start high; keep mask small
    t = np.percentile(L, 96)
    mask = (L >= t).astype(np.uint8) * 255

    # Feathered mask to avoid halos
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=2)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=18)
    mask_f = np.clip(mask_f, 0.0, 1.0)

    # Safety: never let mask cover too much
    if float(np.mean(mask_f)) > 0.18:
        # tighten threshold
        t = np.percentile(L, 98)
        mask = (L >= t).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=18)
        mask_f = np.clip(mask_f, 0.0, 1.0)

    m3 = np.dstack([mask_f, mask_f, mask_f])
    out = fused.astype(np.float32) * (1.0 - m3) + dark.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


def window_pull_strong(fused: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """
    Strong window recovery (aggressive) but SAFE:
    - mask from fused highlights
    - adaptive threshold so mask coverage stays small
    - heavy feather to avoid halos/crunch
    """
    L = luminance_L(fused)

    # We want only real blown areas, not ceiling/walls.
    # Target: mask coverage ~ 5% to 14% (depends on shot)
    target_max = 0.14

    # Start threshold a bit lower than soft, but not crazy.
    t = np.percentile(L, 94)

    # Auto-tighten if mask too big
    for _ in range(6):
        mask = (L >= t).astype(np.uint8) * 255

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((17, 17), np.uint8), iterations=2)
        mask_f = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigmaX=24)
        mask_f = np.clip(mask_f, 0.0, 1.0)

        coverage = float(np.mean(mask_f))
        if coverage <= target_max:
            break

        # mask too big -> tighten threshold (more selective)
        t = min(t + 1.5, 99.2)

    # Final feather again (extra anti-halo)
    mask_f = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=10)
    mask_f = np.clip(mask_f, 0.0, 1.0)

    # Use slightly more dark in the window region (aggressive pull)
    bias = 0.90
    m = mask_f * bias

    m3 = np.dstack([m, m, m])
    out = fused.astype(np.float32) * (1.0 - m3) + dark.astype(np.float32) * m3
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_exposure(img: np.ndarray, target_mid: float = 0.58) -> np.ndarray:
    """
    Fix "too dark overall":
    - higher gain ceiling so it can recover after window pull
    - still prevents blowing out
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0

    med = float(np.median(L))
    med = max(med, 1e-4)

    gain = target_mid / med

    # IMPORTANT: allow more lift than before
    gain = np.clip(gain, 0.75, 2.60)

    L = np.clip(L * gain, 0, 1)

    # mild shadow lift if very dark
    if med < 0.40:
        L = np.power(L, 0.88)

    lab[:, :, 0] = L * 255.0
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def auto_exposure(img: np.ndarray, target_mid: float = 0.58) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] / 255.0

    med = float(np.median(L))
    med = max(med, 1e-4)

    gain = target_mid / med
    gain = np.clip(gain, 0.75, 1.35)
    L = np.clip(L * gain, 0, 1)

    if med < 0.48:
        L = np.power(L, 0.92)  # lift shadows a bit

    lab[:, :, 0] = L * 255.0
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def gentle_contrast(img: np.ndarray, amount: float = 0.10) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]

    base = cv2.GaussianBlur(L, (0, 0), sigmaX=12)
    detail = L - base
    L2 = L + detail * (amount)

    lab[:, :, 0] = np.clip(L2, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

def gentle_sharpen(img: np.ndarray, strength: float = 0.22) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
    out = cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()

def order_by_brightness(images: list) -> list:
    """
    Auto-detect darkest -> brightest.
    Uses median LAB luminance.
    """
    vals = []
    for idx, im in enumerate(images):
        L = luminance_L(im)
        vals.append((float(np.median(L)), idx))
    vals.sort(key=lambda x: x[0])  # low->high
    return [images[i] for _, i in vals]


# -----------------------------
# Main job processor
# -----------------------------
def process_job(payload: dict):
    job_id = payload["job_id"]
    input_urls = payload["input_urls"]
    style = payload.get("style", "natural")
    order = payload.get("order", "as_provided")
    webhook_url = payload.get("webhook_url")
    webhook_secret = payload.get("webhook_secret")

    result_key = f"hdr_result:{job_id}"
    redis_conn.hset(result_key, mapping={"status": "processing"})
    redis_conn.expire(result_key, 3600)

    print(f"[Worker] Processing job {job_id} with {len(input_urls)} images, style={style}, order={order}")
    start_time = time.time()

    try:
        # 1) Download
        print(f"[Worker] Step 1/9: Downloading {len(input_urls)} images...")
        images = [download_image(url) for url in input_urls]
        print(f"[Worker] Downloaded sizes: {[img.shape for img in images]}")

        # 2) Normalize sizes (make sure all same shape)
        ref_h, ref_w = images[len(images)//2].shape[:2]
        images = [cv2.resize(im, (ref_w, ref_h), interpolation=cv2.INTER_LANCZOS4) if im.shape[:2] != (ref_h, ref_w) else im for im in images]

        # 3) Scene detect
        print("[Worker] Step 2/9: Scene detection...")
        scene = detect_scene(images)
        print(f"[Worker] Scene: {scene}")

        # 4) Lens + vignette (safe mild)
        print("[Worker] Step 3/9: Lens + vignette...")
        images = [correct_lens(im) for im in images]
        images = [remove_vignette(im) for im in images]

        # 5) Align
        print("[Worker] Step 4/9: Alignment...")
        images = align_images(images)

        # 6) Ghost reduce
        print("[Worker] Step 5/9: Ghost reduction...")
        images = remove_ghosts(images)

        # 7) Order handling in worker:
        # If user said dark-normal-bright, we will STILL auto-sort for safety.
        # If user said random, keep as is (already shuffled in API).
        if order in ("dark-normal-bright", "as_provided"):
            images = order_by_brightness(images)

        dark_img = images[0]
        mid_img = images[len(images)//2]

        # 8) Better fusion + window pull + normalize + gentle finish
        print("[Worker] Step 6/9: Exposure fusion...")
        fused = exposure_fusion_better(images)

        print("[Worker] Step 7/9: Window pull...")
        fused = window_pull(fused, dark_img, mid_img)

        print("[Worker] Step 8/9: Brightness normalize...")
        target_mid = 0.58 if scene == "interior" else 0.54
        fused = auto_exposure(fused, target_mid=target_mid)

        print("[Worker] Step 9/9: Gentle contrast + sharpen...")
        fused = gentle_contrast(fused, amount=0.10 if scene == "interior" else 0.08)
        fused = gentle_sharpen(fused, strength=0.22)

        elapsed = time.time() - start_time
        print(f"[Worker] Done in {elapsed:.1f}s. Output shape: {fused.shape}")

        # Encode
        result_bytes = encode_jpeg(fused, FINAL_JPEG_QUALITY)
        print(f"[Worker] JPEG size: {len(result_bytes) / 1024:.0f} KB")

        # Webhook
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

        # Store status
        redis_conn.hset(result_key, mapping={"status": "completed", "output_url": "webhook_delivered"})
        redis_conn.expire(result_key, 3600)
        print(f"[Worker] Job {job_id} completed.")

    except Exception as e:
        elapsed = time.time() - start_time
        err = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[Worker] Job {job_id} FAILED after {elapsed:.1f}s:\n{err}")

        redis_conn.hset(result_key, mapping={"status": "failed", "error": str(e)})
        redis_conn.expire(result_key, 3600)

        if webhook_url:
            try:
                headers = {"Content-Type": "application/json"}
                if webhook_secret:
                    headers["x-webhook-secret"] = webhook_secret
                requests.post(
                    webhook_url,
                    json={"jobId": job_id, "status": "failed", "error": str(e)},
                    headers=headers,
                    timeout=30
                )
            except Exception:
                pass

        raise
