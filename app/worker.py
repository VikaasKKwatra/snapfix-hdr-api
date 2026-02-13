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


# -------------------------
# Utilities
# -------------------------

def download_image(url: str) -> np.ndarray:
    resp = requests.get(str(url), timeout=120)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from URL")
    return img


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("Failed to encode JPEG")
    return buf.tobytes()


def detect_scene(images: list) -> str:
    img = images[len(images) // 2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mean_sat = float(np.mean(s))
    bright_pct = float(np.sum(v > 200)) / v.size
    if bright_pct > 0.25 and mean_sat > 60:
        return "exterior"
    return "interior"


# -------------------------
# Bracket sorting (supports random order)
# -------------------------

def sort_brackets_by_exposure(images: list):
    """
    Sort by robust brightness score so we can reliably pick darkest/mid/bright
    even when the upload order is random.
    Returns: dark, mid, bright, sorted_images
    """
    if len(images) < 3:
        # still works with 2, but window pull is limited
        return images[0], images[len(images)//2], images[-1], images

    scores = []
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = float(np.median(gray))  # robust against blown windows
        scores.append((score, i))

    scores_sorted = sorted(scores, key=lambda x: x[0])  # low -> high
    sorted_images = [images[i] for _, i in scores_sorted]

    dark = sorted_images[0]
    bright = sorted_images[-1]
    mid = sorted_images[len(sorted_images)//2]
    return dark, mid, bright, sorted_images


# -------------------------
# Alignment / ghosting (your originals, kept)
# -------------------------

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
            aligned_img = cv2.warpAffine(
                img, warp_matrix, (w, h),
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
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


# -------------------------
# HDR merge (Mertens) - with more real-estate friendly weights
# -------------------------

def exposure_fusion(images: list) -> np.ndarray:
    # These weights reduce color weirdness + reduce crunchy contrast
    merge = cv2.createMergeMertens(
        contrast_weight=0.9,
        saturation_weight=0.55,
        exposure_weight=0.95,
    )
    fusion = merge.process(images)  # float32 0..1-ish
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion


# -------------------------
# Finishing (new: window pull + natural look)
# -------------------------

def balanced_awb(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    a_mean = float(np.mean(A))
    b_mean = float(np.mean(B))
    strength = 0.10  # slightly gentler than your 0.12
    A = A - (a_mean - 128) * strength
    B = B - (b_mean - 128) * strength
    # keep B closer to neutral to prevent yellow/green shifts
    B = np.clip(B, 128 - 7, 128 + 3)
    lab = cv2.merge([L, np.clip(A, 0, 255), np.clip(B, 0, 255)])
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def lift_mids(img: np.ndarray, gamma=0.92, exposure=0.06) -> np.ndarray:
    """
    Brighten interiors without halos (replaces CLAHE).
    gamma < 1 brightens midtones.
    """
    x = img.astype(np.float32) / 255.0
    x = np.clip(x + exposure, 0, 1)
    x = np.power(x, gamma)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def window_pull_blend(fused: np.ndarray, darkest: np.ndarray) -> np.ndarray:
    """
    Softly blend darkest exposure into bright window regions to recover detail
    WITHOUT halos.
    """
    fused_f = fused.astype(np.float32) / 255.0
    dark_f = darkest.astype(np.float32) / 255.0

    # luminance of fused
    lum = 0.2126 * fused_f[:, :, 2] + 0.7152 * fused_f[:, :, 1] + 0.0722 * fused_f[:, :, 0]

    # soft highlight mask (tune start/end)
    start, end = 0.78, 0.94
    mask = (lum - start) / (end - start)
    mask = np.clip(mask, 0.0, 1.0)

    # heavy feathering removes halos
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10)
    mask3 = np.dstack([mask, mask, mask])

    out = fused_f * (1 - mask3) + dark_f * mask3

    # desaturate highlights a bit (prevents neon windows)
    hsv = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1 - 0.20 * mask)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    out2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    return np.clip(out2 * 255, 0, 255).astype(np.uint8)


def mild_denoise(img: np.ndarray, scene: str) -> np.ndarray:
    # interiors benefit from mild denoise; exteriors less
    if scene == "interior":
        return cv2.fastNlMeansDenoisingColored(img, None, 1, 1, 7, 21)
    return img


def safe_sharpen_edges(img: np.ndarray, amount=0.30, radius=1.10) -> np.ndarray:
    """
    Sharpen only where there are real edges (prevents crunchy walls and halos).
    """
    base = img.astype(np.float32)

    blur = cv2.GaussianBlur(base, (0, 0), sigmaX=radius)
    sharp = cv2.addWeighted(base, 1 + amount, blur, -amount, 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    edges = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (0, 0), 1.2)
    edges3 = np.dstack([edges, edges, edges])

    out = base * (1 - edges3) + sharp * edges3
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------
# Webhook helper
# -------------------------

def send_webhook(webhook_url: str, webhook_secret: str, job_id: str, fused: np.ndarray):
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

    resp = requests.post(webhook_url, json=webhook_payload, headers=headers, timeout=120)
    return resp.status_code


# -------------------------
# Main job runner
# -------------------------

def process_job(payload: dict):
    job_id = payload["job_id"]
    input_urls = payload["input_urls"]
    style = payload.get("style", "natural")

    # ✅ NEW: supports fixed vs random order
    order = payload.get("order", "fixed")  # "fixed" or "random"

    webhook_url = payload.get("webhook_url")
    webhook_secret = payload.get("webhook_secret")

    result_key = f"hdr_result:{job_id}"
    redis_conn.hset(result_key, mapping={"status": "processing"})
    redis_conn.expire(result_key, 3600)

    print(f"[Worker] Processing job {job_id} with {len(input_urls)} images, style={style}, order={order}")
    start_time = time.time()

    try:
        # Step 1: Download
        print(f"[Worker] Step 1: Downloading {len(input_urls)} images...")
        images = [download_image(url) for url in input_urls]
        print(f"[Worker] Downloaded sizes: {[img.shape for img in images]}")

        # Step 2: If random, sort brackets so we know which is darkest
        if order == "random":
            dark, mid, bright, images = sort_brackets_by_exposure(images)
            print("[Worker] Order=random -> auto-sorted brackets by exposure.")
        else:
            # fixed: assume first is darkest, last is brightest, mid is middle
            dark = images[0]
            mid = images[len(images)//2]
            bright = images[-1]

        # Step 3: Scene detection (use mid)
        scene = detect_scene(images)
        print(f"[Worker] Scene: {scene}")

        # Step 4: Alignment (better to align before any aggressive geometry changes)
        print("[Worker] Step 2: ECC alignment...")
        images = align_images(images)

        # Re-pick dark/mid/bright after alignment (same indices)
        if order == "random":
            dark, mid, bright, images = sort_brackets_by_exposure(images)
        else:
            dark = images[0]
            mid = images[len(images)//2]
            bright = images[-1]

        # Step 5: Ghost removal (optional)
        if len(images) >= 3:
            print("[Worker] Step 3: Ghost removal...")
            images = remove_ghosts(images)

        # Step 6: Exposure fusion
        print("[Worker] Step 4: Exposure fusion (Mertens)...")
        fused = exposure_fusion(images)

        # Step 7: AWB (gentle)
        print("[Worker] Step 5: Balanced AWB...")
        fused = balanced_awb(fused)

        # Step 8: Brightness lift (fix too-dark interiors without halos)
        print("[Worker] Step 6: Midtone lift (no CLAHE)...")
        if scene == "interior":
            fused = lift_mids(fused, gamma=0.92, exposure=0.06)
        else:
            fused = lift_mids(fused, gamma=0.96, exposure=0.03)

        # Step 9: Window pull (fix blown windows WITHOUT halos/crunch)
        print("[Worker] Step 7: Window pull blend...")
        fused = window_pull_blend(fused, dark)

        # Step 10: Mild denoise
        print("[Worker] Step 8: Mild denoise...")
        fused = mild_denoise(fused, scene)

        # Step 11: Safe edge sharpening (no crunchy walls)
        print("[Worker] Step 9: Safe edge sharpening...")
        fused = safe_sharpen_edges(fused, amount=0.30, radius=1.10)

        elapsed = time.time() - start_time
        print(f"[Worker] Complete in {elapsed:.1f}s. Output shape: {fused.shape}")

        # Encode/store result
        result_bytes = encode_jpeg(fused, FINAL_JPEG_QUALITY)
        print(f"[Worker] Final JPEG size: {len(result_bytes)/1024:.0f} KB")

        # Webhook
        if webhook_url:
            try:
                print(f"[Worker] Sending webhook to {webhook_url}")
                code = send_webhook(webhook_url, webhook_secret, job_id, fused)
                print(f"[Worker] Webhook response: {code}")
            except Exception as e:
                print(f"[Worker] Webhook failed: {e}")

        # Store status (keep same behavior as your API expects)
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
                requests.post(
                    webhook_url,
                    json={"jobId": job_id, "status": "failed", "error": str(e)},
                    headers=headers,
                    timeout=30
                )
            except Exception:
                pass
