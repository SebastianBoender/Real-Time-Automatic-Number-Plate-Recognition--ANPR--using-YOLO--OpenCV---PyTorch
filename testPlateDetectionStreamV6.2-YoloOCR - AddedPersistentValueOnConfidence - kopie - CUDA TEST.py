import time
import cv2
import mss
import torch
import numpy as np
from ultralytics import YOLO


# Config
VEHICLE_DEVICE = 0   # GPU 0
PLATE_DEVICE = 1     # GPU 1
OCR_DEVICE = 1       # GPU 1

#Use_half proved to be slightly beneficial for faster inference and less vram usage but not noticeable
USE_HALF = True

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
VEHICLE_INPUT_SIZE = 640

region = {"top": 209, "left": 147, "width": 1828, "height": 1024}

# Conf thresholds
VEHICLE_CONF = 0.35
PLATE_CONF = 0.35
OCR_CONF = 0.25

# Cache
CACHE_TTL = 6.0
CACHE_CONFIDENCE_THRESHOLD = 0.80
PRINT_TIMERS_EVERY_N_FRAMES = 10

# Vehicle filtering (necessary after OCR addition to reduce workload, might not be necessary on higher end GPUs)
MIN_VEHICLE_WIDTH = 80
MIN_VEHICLE_HEIGHT = 80

# Add padding because plate model was trained on cropped vehicles (0.08 recommended for now but further testing necessary)
VEHICLE_PAD_X_RATIO = 0.08
VEHICLE_PAD_Y_RATIO = 0.08

# OCR rerun behavior
PLATE_MOVE_TOLERANCE = 12  # pixels in resized frame
OCR_RECHECK_INTERVAL = 2.0  # seconds

# Optional allowed vehicle classes.
# Example: {"car", "truck", "bus"}, however decided to deactivate to reduce workload
ALLOWED_VEHICLE_LABELS = None

plate_cache = {}


# Initialization/environment checks

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")


## 2 GPUs expected because of local SLI setup. Currently GPU 0 is responsible for vehicle detection while GPU 1 is responsible for plate detection + OCR

gpu_count = torch.cuda.device_count()
if gpu_count < 2:
    raise RuntimeError(f"Expected 2 GPUs, but found {gpu_count}.")

torch.backends.cudnn.benchmark = True

print(f"[INFO] PyTorch: {torch.__version__}")
print(f"[INFO] CUDA build: {torch.version.cuda}")
print(f"[INFO] GPU 0: {torch.cuda.get_device_name(0)}")
print(f"[INFO] GPU 1: {torch.cuda.get_device_name(1)}")
print(f"[INFO] Half precision enabled: {USE_HALF}")



# Load models

vehicle_model = YOLO("roboflowModelVehicleDetectionV2/weights.pt")
plate_model = YOLO("RoboflowModelPlatedetectionV2/weights.pt")
ocr_model = YOLO("roboflowModelOCRV3/weights.pt")

vehicle_model.to(f"cuda:{VEHICLE_DEVICE}")
plate_model.to(f"cuda:{PLATE_DEVICE}")
ocr_model.to(f"cuda:{OCR_DEVICE}")

# Warmup to reduce first-frame hitching
dummy_vehicle = np.zeros((VEHICLE_INPUT_SIZE, VEHICLE_INPUT_SIZE, 3), dtype=np.uint8)
dummy_plate = np.zeros((320, 320, 3), dtype=np.uint8)
dummy_ocr = np.zeros((140, 480, 3), dtype=np.uint8)

with torch.no_grad():
    vehicle_model.predict(dummy_vehicle, device=VEHICLE_DEVICE, half=USE_HALF, verbose=False)
    plate_model.predict(dummy_plate, device=PLATE_DEVICE, half=USE_HALF, verbose=False)
    ocr_model.predict(dummy_ocr, device=OCR_DEVICE, half=USE_HALF, verbose=False)



# Helpers

def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    return x1, y1, x2, y2

def cleanup_cache(cache, ttl):
    now = time.time()
    expired = [k for k, v in cache.items() if (now - v["last_seen"]) > ttl]
    for k in expired:
        del cache[k]

def safe_mean(values):
    return float(np.mean(values)) if values else 0.0

def should_keep_vehicle(label, width, height):
    if width < MIN_VEHICLE_WIDTH or height < MIN_VEHICLE_HEIGHT:
        return False
    if ALLOWED_VEHICLE_LABELS is not None and label not in ALLOWED_VEHICLE_LABELS:
        return False
    return True

def compute_vehicle_key(x1, y1, x2, y2):
    vx_center = (x1 + x2) // 2
    vy_center = (y1 + y2) // 2
    v_width = x2 - x1
    v_height = y2 - y1
    return f"{vx_center//20}_{vy_center//20}_{v_width//20}_{v_height//20}"

def expand_vehicle_box(x1, y1, x2, y2, width, height):
    pad_x = int((x2 - x1) * VEHICLE_PAD_X_RATIO)
    pad_y = int((y2 - y1) * VEHICLE_PAD_Y_RATIO)
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(width - 1, x2 + pad_x)
    ey2 = min(height - 1, y2 + pad_y)
    return ex1, ey1, ex2, ey2

def plate_box_moved(old_box, new_box, tolerance):
    if old_box is None:
        return True
    ox1, oy1, ox2, oy2 = old_box
    nx1, ny1, nx2, ny2 = new_box
    return (
        abs(ox1 - nx1) > tolerance or
        abs(oy1 - ny1) > tolerance or
        abs(ox2 - nx2) > tolerance or
        abs(oy2 - ny2) > tolerance
    )




# Main loop

frame_idx = 0

with mss.mss() as sct:
    while True:
        frame_loop_start = time.perf_counter()

        capture_time = 0.0
        preprocess_time = 0.0
        vehicle_time = 0.0
        plate_time = 0.0
        ocr_time = 0.0
        draw_time = 0.0
        cache_cleanup_time = 0.0

        plate_calls = 0
        ocr_calls = 0
        cache_hits = 0
        vehicles_found = 0
        plates_found = 0
        vehicles_kept = 0

        # Cache cleanup

        t0 = time.perf_counter()
        cleanup_cache(plate_cache, CACHE_TTL)
        t1 = time.perf_counter()
        cache_cleanup_time += (t1 - t0)

 
        # Capture

        t0 = time.perf_counter()
        screenshot = sct.grab(region)
        frame = np.array(screenshot)
        t1 = time.perf_counter()
        capture_time += (t1 - t0)


        # CPU preprocessing

        t0 = time.perf_counter()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        orig_height, orig_width = frame.shape[:2]
        frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
        vehicle_input = cv2.resize(
            frame_resized,
            (VEHICLE_INPUT_SIZE, VEHICLE_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        scale_x = TARGET_WIDTH / VEHICLE_INPUT_SIZE
        scale_y = TARGET_HEIGHT / VEHICLE_INPUT_SIZE
        scale_fx = orig_width / TARGET_WIDTH
        scale_fy = orig_height / TARGET_HEIGHT
        t1 = time.perf_counter()
        preprocess_time += (t1 - t0)


        # Vehicle detection on GPU 0

        t0 = time.perf_counter()
        with torch.no_grad():
            vehicle_results = vehicle_model.predict(
                source=vehicle_input,
                verbose=False,
                device=VEHICLE_DEVICE,
                half=USE_HALF,
                conf=VEHICLE_CONF
            )
        t1 = time.perf_counter()
        vehicle_time += (t1 - t0)

        if vehicle_results and vehicle_results[0].boxes is not None:
            vehicles_found = len(vehicle_results[0].boxes)

        if not vehicle_results or vehicle_results[0].boxes is None or len(vehicle_results[0].boxes) == 0:
            total_frame_time = time.perf_counter() - frame_loop_start
            fps = 1.0 / total_frame_time if total_frame_time > 0 else 0.0

            if frame_idx % PRINT_TIMERS_EVERY_N_FRAMES == 0:
                print(
                    f"[TIMERS] frame={frame_idx} | "
                    f"cleanup={cache_cleanup_time:.4f}s | "
                    f"capture={capture_time:.4f}s | "
                    f"preprocess={preprocess_time:.4f}s | "
                    f"vehicle={vehicle_time:.4f}s | "
                    f"plate={plate_time:.4f}s | "
                    f"ocr={ocr_time:.4f}s | "
                    f"draw={draw_time:.4f}s | "
                    f"total={total_frame_time:.4f}s | "
                    f"fps={fps:.2f} | "
                    f"vehicles={vehicles_found} | kept={vehicles_kept} | plates={plates_found} | "
                    f"plate_calls={plate_calls} | ocr_calls={ocr_calls} | cache_hits={cache_hits}"
                )

            cv2.imshow("Vehicle + Plate + OCR", frame_resized)
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for v_box, v_cls in zip(vehicle_results[0].boxes.xyxy, vehicle_results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, v_box[:4])
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, TARGET_WIDTH, TARGET_HEIGHT)
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(v_cls.item())
            label = vehicle_model.names[class_id]
            v_width = x2 - x1
            v_height = y2 - y1

            if not should_keep_vehicle(label, v_width, v_height):
                continue

            vehicles_kept += 1

            t0 = time.perf_counter()
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame_resized,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )
            t1 = time.perf_counter()
            draw_time += (t1 - t0)

            # Expand crop slightly to match standardized dimensions of cropped plates dataset (model was trained on crop vehicles)
            vx1, vy1, vx2, vy2 = expand_vehicle_box(x1, y1, x2, y2, TARGET_WIDTH, TARGET_HEIGHT)
            if vx2 <= vx1 or vy2 <= vy1:
                continue

            vehicle_crop = frame_resized[vy1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue

            vehicle_key = compute_vehicle_key(x1, y1, x2, y2)
            cached_result = plate_cache.get(vehicle_key)
            now = time.time()

     
            # Plate detection on GPU 1
   
            plate_calls += 1
            t0 = time.perf_counter()
            with torch.no_grad():
                plate_results = plate_model.predict(
                    source=vehicle_crop,
                    verbose=False,
                    device=PLATE_DEVICE,
                    half=USE_HALF,
                    conf=PLATE_CONF
                )
            t1 = time.perf_counter()
            plate_time += (t1 - t0)

            if not plate_results or plate_results[0].boxes is None or len(plate_results[0].boxes) == 0:
                if cached_result is not None:
                    cached_result["last_seen"] = now
                continue

            # Pick the best plate detection by confidence if available
            boxes = plate_results[0].boxes
            if hasattr(boxes, "conf") and boxes.conf is not None and len(boxes.conf) > 0:
                best_idx = int(torch.argmax(boxes.conf).item())
            else:
                best_idx = 0

            p_box = boxes.xyxy[best_idx]
            px1, py1, px2, py2 = map(int, p_box[:4])

            # Convert from padded vehicle crop coords back to resized frame coords
            abs_px1 = vx1 + px1
            abs_py1 = vy1 + py1
            abs_px2 = vx1 + px2
            abs_py2 = vy1 + py2

            abs_px1, abs_py1, abs_px2, abs_py2 = clamp_box(
                abs_px1, abs_py1, abs_px2, abs_py2, TARGET_WIDTH, TARGET_HEIGHT
            )
            if abs_px2 <= abs_px1 or abs_py2 <= abs_py1:
                if cached_result is not None:
                    cached_result["last_seen"] = now
                continue

            plates_found += 1
            current_plate_box = (abs_px1, abs_py1, abs_px2, abs_py2)

            t0 = time.perf_counter()
            cv2.rectangle(frame_resized, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)
            cv2.putText(
                frame_resized,
                "Plate",
                (abs_px1, max(20, abs_py1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            t1 = time.perf_counter()
            draw_time += (t1 - t0)

            # Update original frame coords for OCR crop
            orig_px1 = int(abs_px1 * scale_fx)
            orig_py1 = int(abs_py1 * scale_fy)
            orig_px2 = int(abs_px2 * scale_fx)
            orig_py2 = int(abs_py2 * scale_fy)

            orig_px1, orig_py1, orig_px2, orig_py2 = clamp_box(
                orig_px1, orig_py1, orig_px2, orig_py2, orig_width, orig_height
            )
            if orig_px2 <= orig_px1 or orig_py2 <= orig_py1:
                if cached_result is not None:
                    cached_result["last_seen"] = now
                continue

            plate_crop = frame[orig_py1:orig_py2, orig_px1:orig_px2]
            if plate_crop.size == 0:
                if cached_result is not None:
                    cached_result["last_seen"] = now
                continue

            plate_w = orig_px2 - orig_px1
            plate_h = orig_py2 - orig_py1
            if plate_w < 40 or plate_h < 15:
                if cached_result is not None:
                    cached_result["last_seen"] = now
                continue

            # decide whether OCR must be rerun
            use_cached = False
            if cached_result is not None:
                moved = plate_box_moved(cached_result.get("plate_box"), current_plate_box, PLATE_MOVE_TOLERANCE)
                fresh_enough = (now - cached_result["ocr_timestamp"]) <= OCR_RECHECK_INTERVAL
                strong_enough = cached_result["confidence"] >= CACHE_CONFIDENCE_THRESHOLD

                cached_result["last_seen"] = now
                cached_result["plate_box"] = current_plate_box

                if strong_enough and not moved and fresh_enough:
                    use_cached = True

            if use_cached:
                cache_hits += 1
                plate_text = cached_result["text"]
                avg_conf = cached_result["confidence"]
            else:
                plate_input = cv2.resize(plate_crop, (480, 140), interpolation=cv2.INTER_LINEAR)


                # OCR on GPU 1 (tried TrOCR & EasyOCR but even though Yolo isn't recommended for OCR, it's still the most accurate after perfecting the dataset. Custom trained TrOCR model should be better though and possibly less laggy)

                ocr_calls += 1
                t0 = time.perf_counter()
                with torch.no_grad():
                    ocr_results = ocr_model.predict(
                        source=plate_input,
                        verbose=False,
                        device=OCR_DEVICE,
                        half=USE_HALF,
                        conf=OCR_CONF
                    )
                t1 = time.perf_counter()
                ocr_time += (t1 - t0)

                if not ocr_results or ocr_results[0].boxes is None or len(ocr_results[0].boxes) == 0:

                    # If OCR fails but cache exists, fall back to cached text
                    
                    if cached_result is not None and cached_result["confidence"] >= CACHE_CONFIDENCE_THRESHOLD:
                        plate_text = cached_result["text"]
                        avg_conf = cached_result["confidence"]
                        cache_hits += 1
                    else:
                        continue
                else:
                    ocr_chars = []
                    confidences = []

                    for ocr_box, ocr_cls, ocr_conf in zip(
                        ocr_results[0].boxes.xyxy,
                        ocr_results[0].boxes.cls,
                        ocr_results[0].boxes.conf
                    ):
                        char = ocr_model.names[int(ocr_cls.item())]
                        x_coord = float(ocr_box[0].item())
                        conf_score = float(ocr_conf.item())
                        ocr_chars.append((x_coord, char))
                        confidences.append(conf_score)

                    ocr_chars_sorted = sorted(ocr_chars, key=lambda x: x[0])
                    plate_text = ''.join(char for _, char in ocr_chars_sorted)
                    avg_conf = safe_mean(confidences)

                    plate_cache[vehicle_key] = {
                        "text": plate_text,
                        "confidence": avg_conf,
                        "ocr_timestamp": now,
                        "last_seen": now,
                        "plate_box": current_plate_box
                    }

            text_to_show = f"OCR: {plate_text} ({avg_conf:.2f})"
            tx, ty = abs_px1, abs_py2 + 25

            (text_width, text_height), _ = cv2.getTextSize(
                text_to_show,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            y1_bg = max(0, ty - text_height - 5)
            y2_bg = min(TARGET_HEIGHT - 1, ty + 5)
            x1_bg = max(0, tx - 2)
            x2_bg = min(TARGET_WIDTH - 1, tx + text_width + 2)

            t0 = time.perf_counter()
            cv2.rectangle(frame_resized, (x1_bg, y1_bg), (x2_bg, y2_bg), (0, 0, 0), -1)
            cv2.putText(
                frame_resized,
                text_to_show,
                (tx, min(TARGET_HEIGHT - 5, ty)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            t1 = time.perf_counter()
            draw_time += (t1 - t0)

        total_frame_time = time.perf_counter() - frame_loop_start
        fps = 1.0 / total_frame_time if total_frame_time > 0 else 0.0

        #Debugger, can be disabled to reduce noise

        if frame_idx % PRINT_TIMERS_EVERY_N_FRAMES == 0:
            print(
                f"[TIMERS] frame={frame_idx} | "
                f"cleanup={cache_cleanup_time:.4f}s | "
                f"capture={capture_time:.4f}s | "
                f"preprocess={preprocess_time:.4f}s | "
                f"vehicle={vehicle_time:.4f}s | "
                f"plate={plate_time:.4f}s | "
                f"ocr={ocr_time:.4f}s | "
                f"draw={draw_time:.4f}s | "
                f"total={total_frame_time:.4f}s | "
                f"fps={fps:.2f} | "
                f"vehicles={vehicles_found} | kept={vehicles_kept} | plates={plates_found} | "
                f"plate_calls={plate_calls} | ocr_calls={ocr_calls} | cache_hits={cache_hits}"
            )

        cv2.imshow("Vehicle + Plate + OCR", frame_resized)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()