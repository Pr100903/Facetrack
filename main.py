import cv2
import os
import numpy as np
from retinaface import RetinaFace
from deepface import DeepFace
from scipy.spatial.distance import cosine
import time
import argparse
from pathlib import Path

# ---------------- CONFIG ---------------- #
DB_PATH = "training"
MODEL_NAME = "ArcFace"   # Best backend for DeepFace
THRESHOLD = 0.60         # Tune between 0.4 - 0.6
DEBUG = True             # Set True to print distance debug info
USE_GPU = False  # GPU disabled for CPU-only environments
DETECTION_SCALE = 0.5    # Reduce for faster detection (0 < scale <= 1)
# Fast-mode defaults (for higher FPS)
FAST_DETECTION_SCALE = 0.35
FAST_DETECTION_INTERVAL = 12
# Minimum face area (w*h) to attempt recognition — skips tiny faces
MIN_FACE_AREA = 1500
# ---------------------------------------- #

print("GPU Available: False (CPU-only mode)")

print("Loading DeepFace model...")
DeepFace.build_model(MODEL_NAME)
print("Model Loaded Successfully ✅")

# --------- LOAD DATABASE EMBEDDINGS --------- #
database = {}

def load_database():
    print("Generating embeddings for database...")
    for file in os.listdir(DB_PATH):
        if file.endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(DB_PATH, file)
            name = os.path.splitext(file)[0]
            # Extract base name before underscore (e.g., "dhruvit_03" -> "dhruvit")
            name = name.split('_')[0].lower()  # Lowercase for consistency

            embedding = DeepFace.represent(
                img_path=path,
                model_name=MODEL_NAME,
                enforce_detection=False
            )[0]["embedding"]

            database[name] = embedding

    print("Database Loaded:", list(database.keys()))

load_database()

if not database:
    print("ERROR: database is empty. Add face images (jpg/png) into the 'db' folder named after the person (e.g. alice.jpg) and rerun.")
    raise SystemExit

# Performance / streaming settings
DETECTION_INTERVAL = 5   # run face detection every N frames
SCALE = 0.6              # resize factor for faster detection (0 < SCALE <= 1)


# Check whether OpenCV has GUI support (cv2.imshow). Some builds (headless)
# don't implement HighGUI functions which causes runtime errors on Windows.
def _check_opencv_gui_available():
    try:
        cv2.namedWindow("__opencv_gui_check__", cv2.WINDOW_NORMAL)
        cv2.imshow("__opencv_gui_check__", np.zeros((2, 2), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("__opencv_gui_check__")
        return True
    except Exception:
        return False

OPENCV_GUI = _check_opencv_gui_available()
if not OPENCV_GUI:
    print("Warning: OpenCV was built without GUI support (cv2.imshow unavailable).")
    print("On Windows install the GUI-enabled wheel: pip uninstall opencv-python-headless -y && pip install opencv-python")


def create_tracker():
    # Prefer KCF (good balance) fall back to MOSSE if not available
    try:
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        elif hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
    except Exception:
        pass

    # Fallback
    try:
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
            return cv2.legacy.TrackerMOSSE_create()
        elif hasattr(cv2, 'TrackerMOSSE_create'):
            return cv2.TrackerMOSSE_create()
    except Exception:
        pass

    return None

# --------- RECOGNITION FUNCTION --------- #
def recognize_face(face_img):
    """
    face_img: numpy array (cropped face from cv2)
    Returns: (name, distance)
    """
    try:
        # DeepFace.represent() can accept numpy arrays directly
        result = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            enforce_detection=False
        )
        embedding = result[0]["embedding"]
    except Exception as e:
        if DEBUG:
            print(f"Error generating embedding: {e}")
        return "Unknown", 1.0

    min_distance = 1.0
    identity = "Unknown"

    for name, db_embedding in database.items():
        distance = cosine(embedding, db_embedding)

        if distance < min_distance:
            min_distance = distance
            identity = name

    if DEBUG:
        print(f"Best match: {identity} (distance={min_distance:.4f}, threshold={THRESHOLD})")

    if min_distance < THRESHOLD:
        return identity, min_distance
    else:
        return "Unknown", min_distance


# --------- START CAMERA --------- #
cap = cv2.VideoCapture(0)

print("Starting Camera... Press 'q' to exit")

trackers = []  # list of dicts: {'tracker': tracker, 'name': name, 'distance': d}
frame_count = 0
fps_samples_live = []  # collect FPS samples for live camera

# --- Command line options ---
parser = argparse.ArgumentParser(description="Face recognition (camera or image folder)")
parser.add_argument("--mode", choices=["camera", "images"], default="camera", help="Run on live camera or on images")
parser.add_argument("--images", help="Path to an image file or a folder of images (used when --mode images)")
parser.add_argument("--delay", type=int, default=0, help="Show image output delay in ms (0 waits for keypress)")
parser.add_argument("--eval", action="store_true", help="Evaluate on testing folder and print accuracy")
parser.add_argument("--fast", action="store_true", help="Enable fast mode (higher FPS, lower accuracy)")
args = parser.parse_args()

# Apply fast-mode overrides if requested
if args.fast:
    DETECTION_SCALE = FAST_DETECTION_SCALE
    DETECTION_INTERVAL = FAST_DETECTION_INTERVAL
    print(f"Fast mode enabled: DETECTION_SCALE={DETECTION_SCALE}, DETECTION_INTERVAL={DETECTION_INTERVAL}")

if args.eval:
    import re
    test_path = Path("testing")
    if not test_path.is_dir():
        print("ERROR: 'testing' folder not found")
        raise SystemExit

    test_imgs = sorted([x for x in test_path.iterdir() if x.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not test_imgs:
        print("No test images found in 'testing' folder")
        raise SystemExit

    # Metrics tracking
    correct = 0
    total = 0
    per_person = {}  # {name: {'correct': count, 'total': count}}
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    
    # Timing metrics
    detection_times = []
    embedding_times = []
    matching_times = []
    fps_samples = []  # per-image FPS during evaluation

    # Lists for ROC/AUC/EER and verification metrics
    scores = []   # higher = more likely genuine (we'll use -distance)
    labels = []   # 1 = genuine (known), 0 = impostor (unknown)

    # Counters for detection/identification
    TP = FP = TN = FN = 0

    for imgp in test_imgs:
        stem = imgp.stem
        expected_name = re.split(r'[_\(\s]', stem)[0].lower()
        
        # Initialize per-person tracking
        if expected_name not in per_person:
            per_person[expected_name] = {'correct': 0, 'total': 0}

        img = cv2.imread(str(imgp))
        if img is None:
            continue

        # DETECTION TIME - Resize for faster detection
        det_start = time.time()
        img_small = cv2.resize(img, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        detections = RetinaFace.detect_faces(img_small)
        detection_times.append(time.time() - det_start)

        recognized = False
        is_unknown = expected_name == "unknown"

        if isinstance(detections, dict):
            for key in detections.keys():
                x1, y1, x2, y2 = detections[key]["facial_area"]
                
                # Scale coordinates back to original image size
                x1 = int(x1 / DETECTION_SCALE)
                y1 = int(y1 / DETECTION_SCALE)
                x2 = int(x2 / DETECTION_SCALE)
                y2 = int(y2 / DETECTION_SCALE)
                
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
                face = img[y1:y2, x1:x2]

                # Skip very small faces (speeds up processing)
                try:
                    area = max(0, (y2 - y1)) * max(0, (x2 - x1))
                except Exception:
                    area = 0
                if area < MIN_FACE_AREA:
                    if DEBUG:
                        print(f"Skipping small face (area={area})")
                    # Treat as not recognized for metrics
                    continue
                
                # EMBEDDING TIME
                emb_start = time.time()
                name, distance = recognize_face(face)
                embedding_times.append(time.time() - emb_start)

                # Compute per-image FPS = 1 / (detection + embedding)
                try:
                    dt_sum = detection_times[-1] + embedding_times[-1]
                    fps_img = 1.0 / dt_sum if dt_sum > 0 else 0.0
                except Exception:
                    fps_img = 0.0
                fps_samples.append(fps_img)

                # MATCHING TIME (included in recognize_face, but tracking separately)
                matching_times.append(time.time() - emb_start)

                # Check if recognition is correct
                is_correct = name.lower() == expected_name.lower()

                # True label: known (in database) or unknown
                true_known = expected_name in database
                pred_known = (name.lower() != "unknown")

                # Update confusion matrix for known-vs-unknown detection
                if true_known and pred_known:
                    TP += 1
                elif true_known and not pred_known:
                    FN += 1
                elif not true_known and pred_known:
                    FP += 1
                else:
                    TN += 1

                # For ROC/AUC use score = -distance (higher => more likely genuine)
                scores.append(-distance)
                labels.append(1 if true_known else 0)

                if is_correct:
                    correct += 1
                    recognized = True
                    if not is_unknown:
                        known_correct += 1
                    else:
                        unknown_correct += 1
                    print(f"✓ {imgp.name}: recognized as {name}")
                else:
                    print(f"✗ {imgp.name}: expected {expected_name}, got {name} (dist={distance:.4f})")

                # Track per-person
                per_person[expected_name]['correct'] += int(is_correct)
                per_person[expected_name]['total'] += 1

                if not is_unknown:
                    known_total += 1
                else:
                    unknown_total += 1

                break
        else:
            # No face detected
            if is_unknown:
                correct += 1
                recognized = True
                unknown_correct += 1
                per_person[expected_name]['correct'] += 1
                print(f"✓ {imgp.name}: correctly detected as unknown")
            else:
                print(f"✗ {imgp.name}: no face detected")
                per_person[expected_name]['total'] += 1
                known_total += 1

        if is_unknown and not recognized:
            unknown_total += 1
            per_person[expected_name]['total'] += 1
            # no detection -> add a negative (impostor) sample with worst score
            scores.append(-1.0)
            labels.append(0)

        total += 1

    # Print detailed evaluation results
    print("\n" + "=" * 50)
    print("--- Evaluation Results ---")
    print("=" * 50)
    print(f"Total Samples Evaluated: {total}")
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Overall Accuracy: {accuracy:.2f} %")

    known_acc = (known_correct / known_total * 100) if known_total > 0 else 0
    unknown_acc = (unknown_correct / unknown_total * 100) if unknown_total > 0 else 0
    print(f"\nKnown Faces Accuracy: {known_acc:.1f} %")
    print(f"Unknown Detection Accuracy: {unknown_acc:.1f} %")

    print("\nPer Person Accuracy:")
    for person in sorted(per_person.keys()):
        stats = per_person[person]
        if stats['total'] > 0:
            person_acc = (stats['correct'] / stats['total'] * 100)
            print(f"{person.capitalize()} : {person_acc:.1f} %")

    # Precision/Recall/F1 for known-vs-unknown detection
    prec = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    rec = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    far = (FP / (FP + TN)) if (FP + TN) > 0 else 0.0

    print(f"\nPrecision (known detection): {prec:.4f}")
    print(f"Recall (known detection): {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"FAR (False Acceptance Rate): {far:.4f}")

    # Compute ROC / AUC / EER from scores and labels
    auc = None
    eer = None
    if len(scores) > 1 and len(set(labels)) > 1:
        # build ROC curve by sweeping thresholds
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)
        # thresholds from max to min
        thresholds = np.sort(np.unique(scores_arr))[::-1]
        tprs = []
        fprs = []
        for thr in thresholds:
            preds = (scores_arr >= thr).astype(int)
            TP_t = int(((preds == 1) & (labels_arr == 1)).sum())
            FP_t = int(((preds == 1) & (labels_arr == 0)).sum())
            FN_t = int(((preds == 0) & (labels_arr == 1)).sum())
            TN_t = int(((preds == 0) & (labels_arr == 0)).sum())
            tpr = TP_t / (TP_t + FN_t) if (TP_t + FN_t) > 0 else 0.0
            fpr = FP_t / (FP_t + TN_t) if (FP_t + TN_t) > 0 else 0.0
            tprs.append(tpr)
            fprs.append(fpr)

        # AUC (trapezoidal rule) on FPR vs TPR
        fprs_arr = np.array(fprs)
        tprs_arr = np.array(tprs)
        # ensure increasing FPR for integration
        sort_idx = np.argsort(fprs_arr)
        x = fprs_arr[sort_idx]
        y = tprs_arr[sort_idx]
        # Use numpy.trapz when available, otherwise fallback to manual trapezoidal rule
        if hasattr(np, 'trapz'):
            auc = np.trapz(y, x)
        else:
            if x.size < 2:
                auc = 0.0
            else:
                dx = x[1:] - x[:-1]
                auc = float(np.sum(dx * (y[1:] + y[:-1]) / 2.0))

        # EER: point where FPR ~= 1 - TPR
        fnrs = 1 - tprs_arr
        abs_diffs = np.abs(fprs_arr - fnrs)
        eer_idx = np.argmin(abs_diffs)
        eer = (fprs_arr[eer_idx] + fnrs[eer_idx]) / 2.0

    if auc is not None:
        print(f"AUC: {auc:.4f}")
    else:
        print("AUC: N/A (insufficient class variety)")
    if eer is not None:
        print(f"EER: {eer:.4f}")
    else:
        print("EER: N/A (insufficient class variety)")

    # Print timing metrics
    print("\n--- Performance Metrics ---")
    if detection_times:
        avg_detection = np.mean(detection_times)
        print(f"Detection Time : {avg_detection:.4f} sec")
    if embedding_times:
        avg_embedding = np.mean(embedding_times)
        print(f"Embedding Time : {avg_embedding:.4f} sec")
    if matching_times:
        avg_matching = np.mean(matching_times)
        print(f"Matching Time : {avg_matching:.4f} sec")
    
    total_time = sum(detection_times) + sum(embedding_times)
    if total_time > 0:
        print(f"Total Time : {total_time:.4f} sec")

    # FPS stats for evaluation (per-image)
    if fps_samples:
        fps_arr = np.array(fps_samples)
        print(f"\nEvaluation FPS - min: {fps_arr.min():.2f}, avg: {fps_arr.mean():.2f}, max: {fps_arr.max():.2f}")

    print(f"\nGPU Acceleration: {'✅ Enabled' if USE_GPU else '❌ Disabled'}")
    print(f"Detection Scale: {DETECTION_SCALE} (Lower = Faster, Less Accurate)")
    print("\n--- Optimization Tips ---")
    print("1. Reduce DETECTION_SCALE further (e.g., 0.3-0.4) for speed")
    # Note: GPU not used in this CPU-only build
    print("3. Batch process multiple images for better GPU utilization")
    print("4. Use lighter detection models if available")
    print("=" * 50)
    raise SystemExit

if args.mode == "images":
    img_path = args.images
    if not img_path:
        print("ERROR: --images must be provided when --mode images")
        raise SystemExit

    p = Path(img_path)
    imgs = []
    if p.is_file():
        imgs = [p]
    elif p.is_dir():
        imgs = sorted([x for x in p.iterdir() if x.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    else:
        print(f"ERROR: image path not found: {img_path}")
        raise SystemExit

    if not imgs:
        print("No images found in the given path.")
        raise SystemExit

    idx = 0
    paused = False
    print("Controls: SPACE=next, B=back, P=pause/resume, Q=quit")

    while True:
        imgp = imgs[idx]
        img = cv2.imread(str(imgp))
        if img is None:
            print(f"Failed to read {imgp}")
            idx = (idx + 1) % len(imgs)
            continue

        detections = RetinaFace.detect_faces(img)
        if isinstance(detections, dict):
            for key in detections.keys():
                x1, y1, x2, y2 = detections[key]["facial_area"]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
                face = img[y1:y2, x1:x2]
                name, distance = recognize_face(face)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{name}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

        # Add status text
        status = f"[{idx+1}/{len(imgs)}] {imgp.name}"
        if paused:
            status += " [PAUSED]"
        cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if OPENCV_GUI:
            cv2.imshow("Image Recognition", img)
        else:
            # Running headless: skip interactive display
            if 'HEADLESS_OUTPUT_DIR' not in globals():
                HEADLESS_OUTPUT_DIR = Path("headless_output")
                HEADLESS_OUTPUT_DIR.mkdir(exist_ok=True)
                print(f"No GUI available — saving preview frames to {HEADLESS_OUTPUT_DIR}")
            out_path = HEADLESS_OUTPUT_DIR / f"img_{idx+1:04d}_{imgp.name}"
            cv2.imwrite(str(out_path), img)
        
        # Determine wait time
        if paused:
            wait_time = 0  # Wait indefinitely when paused
        else:
            wait_time = args.delay if args.delay > 0 else 0

        if OPENCV_GUI:
            k = cv2.waitKey(wait_time)
        else:
            k = -1
        
        # Handle key presses
        if k != -1:
            key_char = chr(k & 0xFF).lower()
            if key_char == 'q':
                break
            elif key_char == ' ':  # SPACE - next image
                idx = (idx + 1) % len(imgs)
            elif key_char == 'b':  # B - back/previous image
                idx = (idx - 1) % len(imgs)
            elif key_char == 'p':  # P - pause/resume
                paused = not paused
        elif args.delay > 0 and not paused:
            # Auto-advance if delay is set and not paused
            idx = (idx + 1) % len(imgs)

    cv2.destroyAllWindows()
    raise SystemExit

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Use tracking to avoid running detection+recognition every frame
    frame_count += 1

    # Update existing trackers first
    new_trackers = []
    for t in trackers:
        tracker = t['tracker']
        if tracker is None:
            continue
        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            x2, y2 = x + w, y + h
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{t['name']}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)
            new_trackers.append(t)

    trackers = new_trackers

    # Perform detection every DETECTION_INTERVAL frames
    if frame_count % DETECTION_INTERVAL == 0:
        # resize for faster detection
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        detections = RetinaFace.detect_faces(small)

        if isinstance(detections, dict):
            trackers = []
            for key in detections.keys():
                x1, y1, x2, y2 = detections[key]["facial_area"]

                # scale coords back to original frame
                x1 = int(x1 / SCALE)
                y1 = int(y1 / SCALE)
                x2 = int(x2 / SCALE)
                y2 = int(y2 / SCALE)

                # clamp coords
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                face = frame[y1:y2, x1:x2]

                # Skip very small faces (speeds up processing)
                try:
                    area = max(0, (y2 - y1)) * max(0, (x2 - x1))
                except Exception:
                    area = 0
                if area < MIN_FACE_AREA:
                    if DEBUG:
                        print(f"Skipping small face in camera (area={area})")
                    name, distance = "Unknown", 1.0
                else:
                    # Recognize once when face first detected
                    name, distance = recognize_face(face)
                if DEBUG:
                    print(f"Recognized: {name} distance={distance:.4f}")

                # Create tracker for this face region
                tracker = create_tracker()
                if tracker is not None:
                    w = x2 - x1
                    h = y2 - y1
                    try:
                        tracker.init(frame, (x1, y1, w, h))
                    except Exception:
                        # some OpenCV builds expect init differently
                        try:
                            tracker.init(frame, (x1, y1, w, h))
                        except Exception:
                            tracker = None

                trackers.append({'tracker': tracker, 'name': name, 'distance': distance})

    dt = time.time() - start_time
    if dt <= 0:
        fps = 0.0
    else:
        fps = 1.0 / dt

    cv2.putText(frame, f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)
    # collect live FPS samples
    try:
        if fps > 0:
            fps_samples_live.append(fps)
    except Exception:
        pass

    if OPENCV_GUI:
        cv2.imshow("RetinaFace + DeepFace POC", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Headless mode: no display. Allow stopping with Ctrl+C in the terminal.
        pass

cap.release()
cv2.destroyAllWindows()
# Print live camera FPS stats if available
if fps_samples_live:
    arr = np.array(fps_samples_live)
    print(f"\nLive Camera FPS - min: {arr.min():.2f}, avg: {arr.mean():.2f}, max: {arr.max():.2f}")
