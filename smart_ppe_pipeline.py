import argparse
import json
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
 
import cv2
from ultralytics import YOLO
 
from gemma_verifier import verify_violation
 
SNAPSHOT_DIR = "snapshots"
LOG_DIR = "logs"
JSONL_PATH = os.path.join(LOG_DIR, "violations_log.jsonl")
LATEST_PATH = os.path.join(LOG_DIR, "latest_violations.json")
LATEST_MAX = 200          # how many recent verdicts to keep in latest_violations.json
REVERIFY_COOLDOWN_SEC = 20  # don't re-send the same still-violating track more often than this
CONTEXT_PAD_FRAC = 0.6      # how much extra space around the person box to include (60% of box size on each side)
 
 
def ensure_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
 
 
def save_context_snapshot(frame, px1, py1, px2, py2, track_id):
    """Crop the person PLUS surrounding context (not a tight crop) so Gemma
    can see nearby machinery, benches, other workers, etc."""
    h_frame, w_frame = frame.shape[:2]
    bw, bh = px2 - px1, py2 - py1
    pad_x, pad_y = int(bw * CONTEXT_PAD_FRAC), int(bh * CONTEXT_PAD_FRAC)
 
    cx1 = max(px1 - pad_x, 0)
    cy1 = max(py1 - pad_y, 0)
    cx2 = min(px2 + pad_x, w_frame)
    cy2 = min(py2 + pad_y, h_frame)
 
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = frame  # fallback to full frame if the crop is degenerate
 
    filename = f"track{track_id}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, crop)
    return path
 
 
def append_jsonl(record: dict):
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
 
 
def update_latest(record: dict):
    recent = deque(maxlen=LATEST_MAX)
    if os.path.exists(LATEST_PATH):
        try:
            with open(LATEST_PATH, "r") as f:
                recent.extend(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    recent.append(record)
    with open(LATEST_PATH, "w") as f:
        json.dump(list(recent), f, indent=2)
 
 
def build_verdict_record(track_id, detector_flag, snapshot_path, llm_result):
    violation_confirmed = llm_result.get("violation_confirmed", False)
    needs_review = llm_result.get("needs_human_review", False)
 
    if needs_review:
        final_status = "NEEDS_REVIEW"
    elif violation_confirmed:
        final_status = "VIOLATION"
    else:
        final_status = "SAFE"
 
    return {
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track_id": track_id,
        "detector_flag": detector_flag,
        "snapshot_path": snapshot_path,
        "llm_verdict": llm_result,
        "final_status": final_status,
    }
 
 
def run_smart_pipeline(video_path, model_path, conf=0.25):
    ensure_dirs()
 
    model = YOLO(model_path)
    names = model.names
    
    PERSON_ID = HELMET_ID = NOHELMET_ID = VEST_ID = None

    for cid, cname in names.items():
        cname = cname.lower()

        if cname == "person":
            PERSON_ID = cid

        elif cname == "head_helmet":
            HELMET_ID = cid

        elif cname == "head_nohelmet":
            NOHELMET_ID = cid

        elif cname == "vest":
            VEST_ID = cid

    print("Person ID :", PERSON_ID)
    print("Helmet ID :", HELMET_ID)
    print("NoHelmet ID :", NOHELMET_ID)
    print("Vest ID :", VEST_ID)
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
 
    # track_id -> {"violating": bool, "last_sent": float}
    track_state = {}
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, verbose=False)[0]
 
        persons, helmets, nohelmets, vests = [], [], [], []
        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == PERSON_ID:
                track_id = int(box.id.item()) if box.id is not None else -1
                persons.append((x1, y1, x2, y2, track_id))
            elif cls == HELMET_ID:
                helmets.append((x1, y1, x2, y2))
            elif cls == VEST_ID:
                vests.append((x1, y1, x2, y2))
 
        for px1, py1, px2, py2, track_id in persons:
            # ---- Helmet check ----
            head_bottom = py1 + int((py2 - py1) * 0.25)
            helmet_found = any(
                px1 <= (hx1 + hx2) // 2 <= px2 and py1 <= (hy1 + hy2) // 2 <= head_bottom
                for hx1, hy1, hx2, hy2 in helmets
            )
 
            # ---- Vest check ----
            torso_top = py1 + int((py2 - py1) * 0.25)
            torso_bottom = py1 + int((py2 - py1) * 0.85)
            vest_found = any(
                px1 <= (vx1 + vx2) // 2 <= px2 and torso_top <= (vy1 + vy2) // 2 <= torso_bottom
                for vx1, vy1, vx2, vy2 in vests
            )
 
            is_violating = not (helmet_found and vest_found)
 
            # ---- Draw (same as before) ----
            helmet_text = "Helmet" if helmet_found else "NO Helmet"
            vest_text = "Vest" if vest_found else "NO Vest"
            color = (0, 255, 0) if (helmet_found and vest_found) else \
                    (0, 255, 255) if (helmet_found or vest_found) else (0, 0, 255)
            label = f"ID {track_id} | {helmet_text} | {vest_text}"
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.putText(frame, label, (px1, max(py1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
 
            # ---- Smart verification layer ----
            if is_violating:
                state = track_state.get(track_id, {"violating": False, "last_sent": 0.0})
                just_started = not state["violating"]
                cooldown_elapsed = (time.time() - state["last_sent"]) > REVERIFY_COOLDOWN_SEC
 
                if just_started or cooldown_elapsed:
                    detector_flag = {
                        "no_helmet": not helmet_found,
                        "no_vest": not vest_found,
                        "track_id": track_id,
                    }
                    snapshot_path = save_context_snapshot(frame, px1, py1, px2, py2, track_id)
                    llm_result = verify_violation(snapshot_path, detector_flag)
                    record = build_verdict_record(track_id, detector_flag, snapshot_path, llm_result)
 
                    append_jsonl(record)
                    update_latest(record)
 
                    print(f"[{record['final_status']}] track {track_id}: {llm_result.get('reasoning', '')}")
 
                    track_state[track_id] = {"violating": True, "last_sent": time.time()}
            else:
                track_state[track_id] = {"violating": False, "last_sent": track_state.get(track_id, {}).get("last_sent", 0.0)}
 
        cv2.imshow("Smart PPE Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart PPE detection: YOLO + Gemma 4 verification layer")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", required=True, help="Path to trained .pt weights (e.g. best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
 
    run_smart_pipeline(args.video, args.model, args.conf)