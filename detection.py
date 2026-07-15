import argparse
import csv
import os
import time
from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


def get_head_box(x1, y1, x2, y2, frac=0.35):
    """Return a box covering roughly the top `frac` of a person's bbox (the head region)."""
    h = y2 - y1
    head_y2 = y1 + int(h * frac)
    return x1, y1, x2, head_y2


class ViolationLogger:
    """
    Handles CSV logging + snapshot saving for PPE violations.
    Logs once per (track_id, violation_type) unless the state clears and
    re-triggers, so you don't get a new row every frame for the same person.
    """

    def __init__(self, out_dir="violations", auto_open=False):
        self.out_dir = out_dir
        self.auto_open = auto_open
        self.snap_dir = os.path.join(out_dir, "snapshots")
        os.makedirs(self.snap_dir, exist_ok=True)

        self.csv_path = os.path.join(out_dir, "violations_log.csv")
        is_new = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, "a", newline="")
        self.writer = csv.writer(self.csv_file)
        if is_new:
            self.writer.writerow(
                ["timestamp", "frame_no", "track_id", "violation_type", "snapshot_path"]
            )

        # track_id -> currently-active violation types (so we don't spam duplicates)
        self.active_violations = defaultdict(set)

    def update(self, frame, frame_no, track_id, missing_helmet, missing_vest):
        current = set()
        if missing_helmet:
            current.add("no_helmet")
        if missing_vest:
            current.add("no_vest")

        new_violations = current - self.active_violations[track_id]
        for v_type in new_violations:
            self._log(frame, frame_no, track_id, v_type)

        self.active_violations[track_id] = current

    def _log(self, frame, frame_no, track_id, v_type):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        fname = f"id{track_id}_{v_type}_{frame_no}.jpg"
        snap_path = os.path.join(self.snap_dir, fname)

        # frame passed in is already fully annotated (all boxes drawn)
        cv2.imwrite(snap_path, frame)

        self.writer.writerow([ts, frame_no, track_id, v_type, snap_path])
        self.csv_file.flush()
        print(f"[VIOLATION] id={track_id} type={v_type} frame={frame_no}")

        if self.auto_open and snap_path:
            self._open_image(snap_path)

    @staticmethod
    def _open_image(path):
        """Opens the snapshot in the OS default image viewer (non-blocking)."""
        try:
            if os.name == "nt":
                os.startfile(path)  # Windows
            elif os.uname().sysname == "Darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception as e:
            print(f"Could not auto-open {path}: {e}")

    def close(self):
        self.csv_file.close()


def run_trained_mode(video_path, model_path, conf=0.25, vote_window=10, vote_thresh=0.6,
                      save_video=None, log_dir="violations", auto_open=False):
    model = YOLO(model_path)
    names = model.names

    PERSON_ID = None
    HARDHAT_ID = None
    VEST_ID = None

    for cid, cname in names.items():
        if cname == "Person":
            PERSON_ID = cid
        elif cname == "Hardhat":
            HARDHAT_ID = cid
        elif cname == "Safety Vest":
            VEST_ID = cid

    if PERSON_ID is None or HARDHAT_ID is None or VEST_ID is None:
        raise ValueError(
            f"Could not resolve required class names from model. "
            f"Got Person={PERSON_ID}, Hardhat={HARDHAT_ID}, Vest={VEST_ID}. "
            f"Check class names in your model/data.yaml match exactly."
        )

    print("Person ID :", PERSON_ID)
    print("Hardhat ID:", HARDHAT_ID)
    print("Vest ID   :", VEST_ID)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_video, fourcc, fps, (w, h))

    logger = ViolationLogger(log_dir, auto_open=auto_open)

    # Rolling per-track history of (helmet_found, vest_found) bools
    helmet_history = defaultdict(lambda: deque(maxlen=vote_window))
    vest_history = defaultdict(lambda: deque(maxlen=vote_window))

    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf,
            verbose=False
        )[0]

        persons = []
        hardhats = []
        vests = []

        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == PERSON_ID:
                track_id = -1
                if box.id is not None:
                    track_id = int(box.id.item())
                persons.append((x1, y1, x2, y2, track_id))
            elif cls == HARDHAT_ID:
                hardhats.append((x1, y1, x2, y2))
            elif cls == VEST_ID:
                vests.append((x1, y1, x2, y2))

        pending_violations = []

        for px1, py1, px2, py2, track_id in persons:

            # Helmet check
            head_bottom = py1 + int((py2 - py1) * 0.25)
            helmet_found = False
            for hx1, hy1, hx2, hy2 in hardhats:
                cx, cy = (hx1 + hx2) // 2, (hy1 + hy2) // 2
                if px1 <= cx <= px2 and py1 <= cy <= head_bottom:
                    helmet_found = True
                    break

            # Vest check
            torso_top = py1 + int((py2 - py1) * 0.25)
            torso_bottom = py1 + int((py2 - py1) * 0.85)
            vest_found = False
            for vx1, vy1, vx2, vy2 in vests:
                cx, cy = (vx1 + vx2) // 2, (vy1 + vy2) // 2
                if px1 <= cx <= px2 and torso_top <= cy <= torso_bottom:
                    vest_found = True
                    break

            # Update rolling history (only meaningful for tracked persons)
            pending_violation = None
            if track_id != -1:
                helmet_history[track_id].append(helmet_found)
                vest_history[track_id].append(vest_found)

                h_hist = helmet_history[track_id]
                v_hist = vest_history[track_id]
                helmet_ratio = sum(h_hist) / len(h_hist)
                vest_ratio = sum(v_hist) / len(v_hist)

                # "Confirmed" status = majority vote over the window
                helmet_confirmed = helmet_ratio >= vote_thresh
                vest_confirmed = vest_ratio >= vote_thresh

                # Only start flagging a violation once we have enough history
                # to trust the vote (avoids false positives on first few frames)
                if len(h_hist) >= min(vote_window, 5):
                    pending_violation = (track_id, not helmet_confirmed, not vest_confirmed)
                helmet_display = helmet_confirmed
                vest_display = vest_confirmed
            else:
                # Untracked person (no ID) -> just show raw per-frame result
                helmet_display = helmet_found
                vest_display = vest_found

            helmet_text = "Helmet" if helmet_display else "NO Helmet"
            vest_text = "Vest" if vest_display else "NO Vest"

            if helmet_display and vest_display:
                color = (0, 255, 0)
            elif helmet_display or vest_display:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            label = f"ID {track_id} | {helmet_text} | {vest_text}"
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.putText(
                frame, label, (px1, max(py1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )

            if pending_violation is not None:
                pending_violations.append(pending_violation)

        # All boxes for this frame are drawn now -> safe to snapshot the full frame
        for track_id, missing_helmet, missing_vest in pending_violations:
            logger.update(frame, frame_no, track_id, missing_helmet, missing_vest)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("PPE Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logger.close()
    print(f"\nViolation log saved to: {logger.csv_path}")
    print(f"Snapshots saved to:     {logger.snap_dir}")


def run_fallback_mode(video_path, conf=0.4):
    """Rough heuristic mode. Uses stock yolov8n.pt (auto-downloads on first run)."""
    model = YOLO("yolov8n.pt")
    PERSON_CLASS_ID = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, classes=[PERSON_CLASS_ID], verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            hx1, hy1, hx2, hy2 = get_head_box(x1, y1, x2, y2)
            head_crop = frame[hy1:hy2, hx1:hx2]
            if head_crop.size == 0:
                continue

            gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size

            hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
            avg_sat = hsv[:, :, 1].mean()

            has_helmet = edge_density < 0.08 and avg_sat > 40
            color = (0, 255, 0) if has_helmet else (0, 0, 255)
            label = "Helmet?" if has_helmet else "No Helmet?"

            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
            cv2.putText(frame, label, (hx1, max(hy1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Helmet Detection (fallback heuristic - not reliable)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet detection on video using YOLOv8 + OpenCV")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--mode", choices=["trained", "fallback"], default="fallback",
                         help="'trained' = use your custom hardhat model, "
                              "'fallback' = quick heuristic demo, no custom model needed")
    parser.add_argument("--model", default=None,
                         help="Path to trained .pt weights (required for --mode trained)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--vote-window", type=int, default=10,
                         help="Frames to average over before confirming a violation (trained mode only)")
    parser.add_argument("--vote-thresh", type=float, default=0.6,
                         help="Fraction of frames in window that must show PPE present to count as 'confirmed' (trained mode only)")
    parser.add_argument("--save-video", default=None,
                         help="Optional path to save annotated output video (trained mode only)")
    parser.add_argument("--log-dir", default="violations",
                         help="Directory to write violations_log.csv and snapshots (trained mode only)")
    parser.add_argument("--auto-open", action="store_true",
                         help="Pop open each violation snapshot in your default image viewer as it happens")
    args = parser.parse_args()

    if args.mode == "trained":
        if not args.model:
            raise ValueError("--model is required when --mode trained (path to your best.pt)")
        run_trained_mode(
            args.video, args.model, args.conf,
            vote_window=args.vote_window,
            vote_thresh=args.vote_thresh,
            save_video=args.save_video,
            log_dir=args.log_dir,
            auto_open=args.auto_open,
        )
    else:
        print("Running in FALLBACK heuristic mode. This is a rough demo, not production-accurate.")
        print("For real accuracy, train a model (see instructions at top of this file) "
              "and run with --mode trained --model path/to/best.pt")
        run_fallback_mode(args.video, args.conf)