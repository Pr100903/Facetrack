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
# ---------------------------------------- #

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

# --- Command line options ---
parser = argparse.ArgumentParser(description="Face recognition (camera or image folder)")
parser.add_argument("--mode", choices=["camera", "images"], default="camera", help="Run on live camera or on images")
parser.add_argument("--images", help="Path to an image file or a folder of images (used when --mode images)")
parser.add_argument("--delay", type=int, default=0, help="Show image output delay in ms (0 waits for keypress)")
parser.add_argument("--eval", action="store_true", help="Evaluate on testing folder and print accuracy")
args = parser.parse_args()

if args.eval:
    test_path = Path("testing")
    if not test_path.is_dir():
        print("ERROR: 'testing' folder not found")
        raise SystemExit

    test_imgs = sorted([x for x in test_path.iterdir() if x.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not test_imgs:
        print("No test images found in 'testing' folder")
        raise SystemExit

    correct = 0
    total = 0

    for imgp in test_imgs:
        # Extract ground truth from filename (handles: "alice_001.jpg", "alice (2).jpg", "alice(2).jpg")
        stem = imgp.stem
        # Remove numbers and parentheses: split by '_', '(', space
        import re
        expected_name = re.split(r'[_\(\s]', stem)[0].lower()

        img = cv2.imread(str(imgp))
        if img is None:
            continue

        detections = RetinaFace.detect_faces(img)
        recognized = False
        if isinstance(detections, dict):
            for key in detections.keys():
                x1, y1, x2, y2 = detections[key]["facial_area"]
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
                face = img[y1:y2, x1:x2]
                name, distance = recognize_face(face)
                if name.lower() == expected_name.lower():
                    correct += 1
                    recognized = True
                    print(f"✓ {imgp.name}: recognized as {name}")
                else:
                    print(f"✗ {imgp.name}: expected {expected_name}, got {name} (dist={distance:.4f})")
                break

        if not recognized:
            print(f"✗ {imgp.name}: no face detected")

        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
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

        cv2.imshow("Image Recognition", img)
        
        # Determine wait time
        if paused:
            wait_time = 0  # Wait indefinitely when paused
        else:
            wait_time = args.delay if args.delay > 0 else 0

        k = cv2.waitKey(wait_time)
        
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

    cv2.imshow("RetinaFace + DeepFace POC", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
