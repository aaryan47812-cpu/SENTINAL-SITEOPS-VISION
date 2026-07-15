import os
import time
import tempfile
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from ppe_backend import load_model, CameraWorker, violation_label

# --------------------------------------------------------------------------
# PAGE CONFIG + THEME
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel SiteOps Vision",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>

    @import url('https://fonts.cdnfonts.com/css/copperplate-gothic-std');

    html, body, [class*="css"], .stApp, p, div, span, h1, h2, h3, h4, h5, h6 {
        font-family: 'Copperplate Gothic Std', sans-serif !important;
    }

    .stApp { background-color: #0b0e14; color: #e6e6e6; }
    section[data-testid="stSidebar"] { display: none; }

    div[data-testid="stSelectbox"] input {
        caret-color: transparent !important;
        pointer-events: none !important;
    }

    .gw-card {
        background-color: #131722;
        border: 1px solid #232838;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .gw-header {
        font-size: 20px;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b93a7;
        margin-bottom: 10px;
    }
    header[data-testid="stHeader"] { display: none !important; }

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        margin-top: 0px !important;
    }

    .gw-title {
        font-size: 35px !important;
        letter-spacing: 5px !important;
        font-weight: 200;
        color: #f2f4f8;
        margin-top: 0px !important;
        line-height: 1 !important;
    }
    .gw-sub {
        font-size: 15px;
        color: #6f7890;
    }
    .gw-alert-box {
        background: linear-gradient(180deg, #2a0f12 0%, #1a0a0c 100%);
        border: 1px solid #7a1f27;
        border-left: 5px solid #ef4444;
        border-radius: 10px;
        padding: 16px 18px;
        animation: gw-pulse 1.4s infinite;
    }
    @keyframes gw-pulse {
        0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.45); }
        70%  { box-shadow: 0 0 0 9px rgba(239,68,68,0); }
        100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    }
    .gw-alert-title {
        color: #ff6b6b;
        font-weight: 800;
        font-size: 15px;
        margin-bottom: 8px;
    }
    .gw-review-box {
        background: linear-gradient(180deg, #2a230f 0%, #1a160a 100%);
        border: 1px solid #7a6a1f;
        border-left: 5px solid #eab308;
        border-radius: 10px;
        padding: 16px 18px;
    }
    .gw-review-title {
        color: #facc15;
        font-weight: 800;
        font-size: 15px;
        margin-bottom: 8px;
    }
    .gw-safe-box {
        background: linear-gradient(180deg, #0e2418 0%, #0a1712 100%);
        border: 1px solid #1f7a45;
        border-left: 5px solid #22c55e;
        border-radius: 10px;
        padding: 22px 18px;
        text-align: center;
    }
    .gw-safe-title {
        color: #4ade80;
        font-weight: 800;
        font-size: 15px;
    }
    .gw-kv { font-size: 13px; margin: 4px 0; color: #d1d5e0; }
    .gw-kv b { color: #f2f4f8; }
    .gw-reasoning {
        font-size: 12px;
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #333a4d;
        color: #9aa2b8;
        font-style: italic;
    }
    .gw-rule {
        font-size: 13px;
        padding: 7px 10px;
        margin-bottom: 6px;
        background-color: #171c2b;
        border-left: 3px solid #3b4462;
        border-radius: 4px;
        color: #c7cce0;
    }
    div[data-testid="stDataFrame"] { border: 1px solid #232838; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# STATIC CONFIG — cameras, zones, rules, PPE requirements
# --------------------------------------------------------------------------
CAMERAS = {
    "CAM-01": {
        "name": "CCTV Cam",
        "zone": "Factory Heavy Machinery",
        "rules": [
            "Helmet mandatory at all times",
            "Safety vest mandatory at all times",
        ],
        "checks": {"helmet", "vest"},
    },
    "CAM-02": {
        "name": "Live Web Cam",
        "zone": "Web Cam",
        "rules": [
            "Mask mandatory at all times",
        ],
        "checks": {"mask"},
    },
}

MODEL_PATH = "best.pt"

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
defaults = {
    "selected_cam": "CAM-01",
    "alert_log": [],
    "active_violations": {},   # cam_id -> violation dict
    "tick": 0,
    "live": True,
    "workers": {},             # cam_id -> CameraWorker | None
    "last_frame": {},          # cam_id -> annotated PIL/np frame
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def placeholder_frame(text="NO SIGNAL — configure a source below"):
    w, h = 900, 480
    img = Image.new("RGB", (w, h), (18, 20, 28))
    draw = ImageDraw.Draw(img)
    for x in range(0, w, 45):
        draw.line([(x, 0), (x, h)], fill=(26, 29, 40), width=1)
    for y in range(0, h, 45):
        draw.line([(0, y), (w, h)], fill=(26, 29, 40), width=1)
    try:
        font = ImageFont.truetype("custom_font.ttf", 18)
    except IOError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) / 2, (h - th) / 2), text, fill=(90, 96, 112), font=font)
    return img


def persist_upload(uploaded_file, cache_key):
    """Write an uploaded video to a temp file once and reuse the path across reruns."""
    if uploaded_file is None:
        return None
    file_id = getattr(uploaded_file, "file_id", None) or f"{uploaded_file.name}-{uploaded_file.size}"
    id_key, path_key = f"{cache_key}_file_id", f"{cache_key}_path"
    if st.session_state.get(id_key) != file_id:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getbuffer())
        tmp.close()
        st.session_state[id_key] = file_id
        st.session_state[path_key] = tmp.name
    return st.session_state[path_key]


# --------------------------------------------------------------------------
# TOP BAR
# --------------------------------------------------------------------------
top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown(
        f"<div class='gw-title'>🛰️ Sentinel SiteOps Vision</div>"
        f"<div class='gw-sub' style='margin-left: 65px;'>Gemma-4 edge reasoning agent · semantic safety monitoring, not raw detection</div>",
        unsafe_allow_html=True,
    )
with top_r:
    st.session_state.live = st.toggle("Live monitoring", value=st.session_state.live)

st.write("")

# --------------------------------------------------------------------------
# CAMERA FEED SETUP
# --------------------------------------------------------------------------
with st.expander("⚙️ Camera Feed Setup", expanded=not any(st.session_state.workers.values())):
    c1, c2, c3 = st.columns([1, 1, 0.6])
    with c1:
        st.markdown("**CAM-01 · CCTV Cam** — Factory Heavy Machinery")
        cam01_file = st.file_uploader(
            "Upload footage (mp4/avi/mov)", type=["mp4", "avi", "mov", "mkv"], key="cam01_uploader"
        )
    with c2:
        st.markdown("**CAM-02 · Live Web Cam**")
        cam02_mode = st.radio(
            "Source", ["Webcam", "Upload video"], key="cam02_mode", horizontal=True, label_visibility="collapsed"
        )
        cam02_file = None
        if cam02_mode == "Upload video":
            cam02_file = st.file_uploader(
                "Upload footage (mp4/avi/mov)", type=["mp4", "avi", "mov", "mkv"], key="cam02_uploader"
            )
    with c3:
        conf = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05, key="conf_slider")

cam01_source = persist_upload(cam01_file, "cam01")
cam02_source = 0 if cam02_mode == "Webcam" else persist_upload(cam02_file, "cam02")
desired_sources = {"CAM-01": cam01_source, "CAM-02": cam02_source}

# --------------------------------------------------------------------------
# WORKER (VideoCapture + model + tracker) LIFECYCLE
# --------------------------------------------------------------------------
for cid, desired in desired_sources.items():
    existing = st.session_state.workers.get(cid)
    if desired is None:
        if existing is not None:
            existing.release()
            st.session_state.workers[cid] = None
        continue
    if existing is None or existing.source != desired:
        if existing is not None:
            existing.release()
        model, class_ids = load_model(cid, MODEL_PATH)  # isolated model instance per camera
        st.session_state.workers[cid] = CameraWorker(cid, desired, model, class_ids, conf=conf)
    else:
        existing.conf = conf

# --------------------------------------------------------------------------
# TICK — read + process one frame per camera, drain async Gemma verdicts
# --------------------------------------------------------------------------
st.session_state.tick += 1

if st.session_state.live:
    for cid, worker in st.session_state.workers.items():
        if worker is None or not worker.is_open:
            continue

        frame = worker.read_frame()
        if frame is not None:
            annotated, resolved_tids = worker.process(frame, CAMERAS[cid]["checks"])
            st.session_state.last_frame[cid] = annotated

            for tid in resolved_tids:
                av = st.session_state.active_violations.get(cid)
                if av and av.get("track_id") == tid:
                    resolved = av.copy()
                    resolved["status"] = "Resolved"
                    resolved["duration_s"] = int(time.time() - av.get("first_seen", time.time()))
                    st.session_state.alert_log.insert(0, resolved)
                    del st.session_state.active_violations[cid]

        for rec in worker.drain_results():
            verdict = rec["llm_verdict"]
            if verdict.get("needs_human_review"):
                status = "Needs Review"
            elif verdict.get("violation_confirmed"):
                status = "Active"
            else:
                continue  # Gemma caught a false alarm -- nothing to surface

            entry = {
                "time": rec["time"],
                "worker_id": f"T-{rec['track_id']}",
                "track_id": rec["track_id"],
                "zone": CAMERAS[cid]["zone"],
                "violation": violation_label(rec["detector_flag"]),
                "confidence": verdict.get("confidence", 0.0),
                "reasoning": verdict.get("reasoning", ""),
                "duration_s": 0,
                "status": status,
                "cam_id": cid,
                "first_seen": time.time(),
            }
            st.session_state.active_violations[cid] = entry

st.session_state.alert_log = st.session_state.alert_log[:40]

sel = st.session_state.selected_cam
cfg = CAMERAS[sel]

# --------------------------------------------------------------------------
# MAIN ROW: camera switcher | live feed | active alert
# --------------------------------------------------------------------------
col_cams, col_feed, col_alert = st.columns([0.8, 3.0, 1.2], gap="medium")

with col_cams:
    st.markdown("<div class='gw-header'>CCTV Cameras</div>", unsafe_allow_html=True)

    cam_ids = list(CAMERAS.keys())
    current_idx = cam_ids.index(st.session_state.selected_cam)

    selected_cid = st.selectbox(
        "Select Camera",
        options=cam_ids,
        index=current_idx,
        format_func=lambda x: f"{x} · {CAMERAS[x]['name']}",
        label_visibility="collapsed",
        key="cam_dropdown",
    )

    if selected_cid != st.session_state.selected_cam:
        st.session_state.selected_cam = selected_cid
        st.rerun()

with col_feed:
    st.markdown(
        f"<div class='gw-header'>Live Feed — {sel} · {cfg['zone']}</div>",
        unsafe_allow_html=True,
    )
    frame = st.session_state.last_frame.get(sel)
    if frame is not None:
        st.image(frame, use_container_width=True, channels="BGR")
    else:
        st.image(placeholder_frame(), use_container_width=True)

    rules_html = f"<div class='gw-card'><div class='gw-header'>Zone Rules — {cfg['zone']}</div>"
    for r in cfg["rules"]:
        rules_html += f"<div class='gw-rule'>▸ {r}</div>"
    rules_html += "</div>"
    st.markdown(rules_html, unsafe_allow_html=True)

with col_alert:
    st.markdown("<div class='gw-header' style='margin-left:2px;'>Active Alert</div>", unsafe_allow_html=True)
    v = st.session_state.active_violations.get(sel)
    if v:
        duration = int(time.time() - v.get("first_seen", time.time()))
        box_class = "gw-review-box" if v["status"] == "Needs Review" else "gw-alert-box"
        title_class = "gw-review-title" if v["status"] == "Needs Review" else "gw-alert-title"
        title_text = "⏳ NEEDS HUMAN REVIEW" if v["status"] == "Needs Review" else "⚠ SAFETY VIOLATION — GEMMA CONFIRMED"
        st.markdown(
            f"""
            <div class='{box_class}'>
                <div class='{title_class}'>{title_text}</div>
                <div class='gw-kv'><b>Violation:</b> {v['violation']}</div>
                <div class='gw-kv'><b>Track ID:</b> {v['worker_id']}</div>
                <div class='gw-kv'><b>Zone:</b> {v['zone']}</div>
                <div class='gw-kv'><b>Confidence:</b> {v['confidence']:.0%}</div>
                <div class='gw-kv'><b>Duration:</b> {duration}s</div>
                <div class='gw-kv'><b>Detected:</b> {v['time']}</div>
                <div class='gw-reasoning'>"{v['reasoning']}"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class='gw-safe-box'>
                <div class='gw-safe-title'>✓ No Active Violation</div>
                <div class='gw-sub' style='margin-top:6px;'>This zone is currently compliant</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    n_active = len(st.session_state.active_violations)
    n_online = sum(1 for w in st.session_state.workers.values() if w is not None and w.is_open)
    st.markdown(
        f"<div class='gw-card'><div class='gw-header'>Site Overview</div>"
        f"<div class='gw-kv'><b>Cameras online:</b> {n_online}/{len(CAMERAS)}</div>"
        f"<div class='gw-kv'><b>Active violations (site-wide):</b> {n_active}</div>"
        f"<div class='gw-kv'><b>Logged today:</b> {len(st.session_state.alert_log)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# BOTTOM: recent alerts table
# --------------------------------------------------------------------------
st.write("")
pad_l, table_col, pad_r = st.columns([0.4, 3.2, 0.4])
with table_col:
    st.markdown("<div class='gw-card'>", unsafe_allow_html=True)
    st.markdown("<div class='gw-header'>Recent Alerts</div>", unsafe_allow_html=True)

    rows = list(st.session_state.active_violations.values()) + st.session_state.alert_log

    if rows:
        table_data = [
            {
                "Time": r["time"],
                "Cam": r.get("cam_id", ""),
                "Track ID": r["worker_id"],
                "Zone": r["zone"],
                "Violation": r["violation"],
                "Confidence": f"{r.get('confidence', 0):.0%}" if "confidence" in r else "—",
                "Duration": f"{r['duration_s']}s",
                "Status": r["status"],
            }
            for r in rows[:20]
        ]
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.markdown("<div class='gw-sub'>No alerts logged yet this session.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# LIVE REFRESH LOOP
# --------------------------------------------------------------------------
if st.session_state.live and any(w is not None and w.is_open for w in st.session_state.workers.values()):
    time.sleep(0.05)
    st.rerun()
