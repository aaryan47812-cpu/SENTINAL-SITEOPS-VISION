"""
Backend glue between the trained YOLO PPE model + ByteTrack + Gemma-4
verification layer, and the Streamlit frontend (app.py).

Design notes
------------
- Each camera gets its OWN YOLO model instance (even though both load the
  same best.pt weights). Ultralytics keeps ByteTrack state attached to the
  model/predictor object, so sharing one model between two independent
  video streams would corrupt both cameras' track IDs.
- Gemma verification (via gemma_verifier.verify_violation -> local Ollama)
  is slow (seconds), so it's dispatched on a background thread per
  detection event. Streamlit's main thread never blocks on it -- it just
  drains a thread-safe queue of finished verdicts on every rerun.
"""

import os
import time
import uuid
import threading
import queue

import cv2
import streamlit as st
from ultralytics import YOLO

from gemma_verifier import verify_violation

SNAPSHOT_DIR = "snapshots"
REVERIFY_COOLDOWN_SEC = 20   # don't re-send the same still-violating track more often than this
CONTEXT_PAD_FRAC = 0.6       # extra context around the person box sent to Gemma

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

CHECK_LABELS = {
    "helmet": ("Helmet", "NO Helmet"),
    "vest": ("Vest", "NO Vest"),
    "mask": ("Mask", "NO Mask"),
}


@st.cache_resource(show_spinner="Loading PPE model...")
def load_model(cache_key: str, model_path: str = "best.pt"):
    """Load an isolated YOLO instance for a given camera (cache_key = cam_id).
    Isolated instances keep ByteTrack state from bleeding between cameras."""
    model = YOLO(model_path)
    names = model.names
    class_ids = {
    "person": None,
    "helmet": None,
    "nohelmet": None,
    "vest": None,
    "mask": None,
    "nomask": None,
}

    for cid, cname in names.items():
        cname = cname.lower()

        if cname == "person":
            class_ids["person"] = cid

        elif cname == "head_helmet":
            class_ids["helmet"] = cid

        elif cname == "head_nohelmet":
            class_ids["nohelmet"] = cid

        elif cname == "vest":
            class_ids["vest"] = cid

        elif cname == "face_mask":
            class_ids["mask"] = cid

        elif cname == "face_nomask":
            class_ids["nomask"] = cid

    print(class_ids)
    return model, class_ids


class CameraWorker:
    """Owns a VideoCapture + tracker state + async Gemma verification queue
    for a single camera feed."""

    def __init__(self, cam_id, source, model, class_ids, conf=0.25):
        self.cam_id = cam_id
        self.source = source
        self.model = model
        self.class_ids = class_ids
        self.conf = conf
        self.cap = cv2.VideoCapture(source)
        print(f"Camera {cam_id} source = {source}")
        print("Opened:", self.cap.isOpened())
        self.track_state = {}        # track_id -> {"violating": bool, "last_sent": float}
        self.awaiting = set()        # track_ids with a verification request in flight
        self.result_queue = queue.Queue()
        self._lock = threading.Lock()

    @property
    def is_open(self):
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self):
        if not self.is_open:
            return None
        ok, frame = self.cap.read()
        if not ok:
            # loop finished video files back to the start; webcams just keep failing if disconnected
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None
        return frame

    def process(self, frame, required_checks):
        """Run detection+tracking on one frame, draw overlays, and dispatch
        async Gemma verification for newly-flagged tracks.
        required_checks: subset of {"helmet", "vest", "mask"} this camera enforces.
        Returns (annotated_frame, resolved_track_ids) -- resolved_track_ids are
        tracks that were violating and just became PPE-compliant this frame."""
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", conf=self.conf, verbose=False
        )[0]

        resolved_tids = []
        persons, helmets, nohelmets, vests, masks, nomasks = [], [], [], [], [], []
        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == self.class_ids["person"]:
                tid = int(box.id.item()) if box.id is not None else -1
                persons.append((x1, y1, x2, y2, tid))
            elif cls == self.class_ids["helmet"]:
                helmets.append((x1, y1, x2, y2))

            elif cls == self.class_ids["nohelmet"]:
                nohelmets.append((x1, y1, x2, y2))

            elif cls == self.class_ids["vest"]:
                vests.append((x1, y1, x2, y2))

            elif cls == self.class_ids["mask"]:
                masks.append((x1, y1, x2, y2))

            elif cls == self.class_ids["nomask"]:
                nomasks.append((x1, y1, x2, y2))

        for px1, py1, px2, py2, tid in persons:
            found = {}

            if "helmet" in required_checks:
                head_bottom = py1 + int((py2 - py1) * 0.25)
                found["helmet"] = any(
                    px1 <= (hx1 + hx2) // 2 <= px2 and py1 <= (hy1 + hy2) // 2 <= head_bottom
                    for hx1, hy1, hx2, hy2 in helmets
                )

            if "vest" in required_checks:
                torso_top = py1 + int((py2 - py1) * 0.25)
                torso_bottom = py1 + int((py2 - py1) * 0.85)
                found["vest"] = any(
                    px1 <= (vx1 + vx2) // 2 <= px2 and torso_top <= (vy1 + vy2) // 2 <= torso_bottom
                    for vx1, vy1, vx2, vy2 in vests
                )

            if "mask" in required_checks:
                head_bottom = py1 + int((py2 - py1) * 0.35)

                mask_found = any(
                    px1 <= (mx1 + mx2)//2 <= px2 and
                    py1 <= (my1 + my2)//2 <= head_bottom
                    for mx1, my1, mx2, my2 in masks
                    )

                nomask_found = any(
                    px1 <= (nx1 + nx2)//2 <= px2 and
                    py1 <= (ny1 + ny2)//2 <= head_bottom
                    for nx1, ny1, nx2, ny2 in nomasks
                )

            if mask_found:
                found["mask"] = True
            elif nomask_found:
                found["mask"] = False
            else:
                found["mask"] = False

            is_violating = any(not v for v in found.values())

            # ---- draw box + label ----
            parts = [f"ID {tid}"]
            for check, present in found.items():
                on, off = CHECK_LABELS[check]
                parts.append(on if present else off)
            label = " | ".join(parts)

            if not found:
                color = (150, 150, 150)
            elif is_violating:
                color = (0, 0, 255) if not any(found.values()) else (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            cv2.putText(frame, label, (px1, max(py1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ---- async verification dispatch ----
            if is_violating and tid != -1:
                state = self.track_state.get(tid, {"violating": False, "last_sent": 0.0})
                just_started = not state["violating"]
                cooldown_elapsed = (time.time() - state["last_sent"]) > REVERIFY_COOLDOWN_SEC
                if (just_started or cooldown_elapsed) and tid not in self.awaiting:
                    self._dispatch_verification(frame, px1, py1, px2, py2, tid, found)
                    self.track_state[tid] = {"violating": True, "last_sent": time.time()}
            elif tid != -1:
                was_violating = self.track_state.get(tid, {}).get("violating", False)
                if was_violating:
                    resolved_tids.append(tid)
                self.track_state[tid] = {
                    "violating": False,
                    "last_sent": self.track_state.get(tid, {}).get("last_sent", 0.0),
                }

        return frame, resolved_tids

    def _dispatch_verification(self, frame, px1, py1, px2, py2, tid, found):
        h_frame, w_frame = frame.shape[:2]
        bw, bh = px2 - px1, py2 - py1
        pad_x, pad_y = int(bw * CONTEXT_PAD_FRAC), int(bh * CONTEXT_PAD_FRAC)
        cx1, cy1 = max(px1 - pad_x, 0), max(py1 - pad_y, 0)
        cx2, cy2 = min(px2 + pad_x, w_frame), min(py2 + pad_y, h_frame)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            crop = frame

        snap_path = os.path.join(SNAPSHOT_DIR, f"{self.cam_id}_track{tid}_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(snap_path, crop)

        detector_flag = {f"no_{k}": (not v) for k, v in found.items()}
        detector_flag["track_id"] = tid

        self.awaiting.add(tid)

        def worker():
            result = verify_violation(snap_path, detector_flag)
            self.result_queue.put({
                "cam_id": self.cam_id,
                "track_id": tid,
                "snapshot": snap_path,
                "detector_flag": detector_flag,
                "llm_verdict": result,
                "time": time.strftime("%H:%M:%S"),
            })
            self.awaiting.discard(tid)

        threading.Thread(target=worker, daemon=True).start()

    def drain_results(self):
        out = []
        while True:
            try:
                out.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return out

    def release(self):
        if self.cap is not None:
            self.cap.release()


def violation_label(detector_flag):
    """Human-readable violation string from a detector_flag dict, e.g. 'Helmet Not Worn'."""
    names = {"no_helmet": "Helmet", "no_vest": "Vest", "no_mask": "Mask"}
    missing = [names[k] for k, v in detector_flag.items() if k in names and v]
    if not missing:
        return "PPE Violation"
    return " & ".join(missing) + " Not Worn"
