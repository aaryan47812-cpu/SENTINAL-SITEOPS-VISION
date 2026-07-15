
import random
import time
from datetime import datetime

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


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

    /* Force the font globally across all Streamlit classes and custom classes */
    html, body, [class*="css"], .stApp, p, div, span, h1, h2, h3, h4, h5, h6 { 
        font-family: 'Copperplate Gothic Std', sans-serif !important; 
    }

    .stApp { background-color: #0b0e14; color: #e6e6e6; }
    section[data-testid="stSidebar"] { display: none; }


    /* Neutralize Selectbox Typing/Search */
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
    /* 1. Nuke the native Streamlit header to clear the collision zone */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* 2. Recalibrate the main container padding */
    .block-container {
        padding-top: 2rem !important; /* Increased from 1rem to clear the browser edge safely */
        padding-bottom: 1rem !important;
        margin-top: 0px !important;
    }

    /* 3. Keep the title margins stripped */
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
    .gw-rule {
        font-size: 13px;
        padding: 7px 10px;
        margin-bottom: 6px;
        background-color: #171c2b;
        border-left: 3px solid #3b4462;
        border-radius: 4px;
        color: #c7cce0;
    }
    .gw-cam-btn {
        font-size: 13px;
        padding: 8px 10px;
        border-radius: 8px;
        margin-bottom: 6px;
    }
    .status-dot {
        height: 8px; width: 8px; border-radius: 50%;
        display: inline-block; margin-right: 6px;
    }
    .badge-critical { background-color:#3a0d10; color:#ff8080; padding:2px 8px; border-radius:5px; font-size:11px; font-weight:700; }
    .badge-resolved { background-color:#0d2b1a; color:#6fe0a0; padding:2px 8px; border-radius:5px; font-size:11px; font-weight:700; }
    .badge-active   { background-color:#3a2b0d; color:#ffcf70; padding:2px 8px; border-radius:5px; font-size:11px; font-weight:700; }

    div[data-testid="stDataFrame"] { border: 1px solid #232838; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

CAMERAS = {
    "CAM-01": {
        "name": "CCTV Cam",
        "zone": "Factory Heavy Machinery",
        "rules": [
            "Helmet mandatory at all times",
            "Safety vest mandatory at all times",
        ],
        "bg": (34, 40, 49),
    },
    "CAM-02": {
        "name": "Live Web Cam",
        "zone": "Web Cam",
        "rules": [
            "Mask Mandatory at all time",
        ],
        "bg": (28, 44, 46),
    },
    
}

VIOLATION_TYPES = {
    "CAM-01": "Helmet Not Worn",
    "CAM-02": "Mask Not worn"
   
}


if "selected_cam" not in st.session_state:
    st.session_state.selected_cam = "CAM-01"
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "active_violations" not in st.session_state:
    st.session_state.active_violations = {}  
if "tick" not in st.session_state:
    st.session_state.tick = 0
if "live" not in st.session_state:
    st.session_state.live = True


def worker_id():
    return f"W-{random.randint(1000, 1099)}"


def maybe_update_violation(cam_id):
    """Simulate Gemma-4's reasoning output ticking forward each refresh."""
    existing = st.session_state.active_violations.get(cam_id)
    if existing:
        # 55% chance the violation persists/escalates, else it resolves
        if random.random() < 0.45:
            existing["duration_s"] += random.randint(3, 6)
            return
        else:
            resolved = existing.copy()
            resolved["status"] = "Resolved"
            st.session_state.alert_log.insert(0, resolved)
            del st.session_state.active_violations[cam_id]
            return
    # No active violation currently — small chance one starts
    if random.random() < 0.18:
        v = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "worker_id": worker_id(),
            "zone": CAMERAS[cam_id]["zone"],
            "violation": VIOLATION_TYPES[cam_id],
            "duration_s": random.randint(3, 8),
            "threshold_s": 10,
            "status": "Active",
            "cam_id": cam_id,
        }
        st.session_state.active_violations[cam_id] = v


def render_frame(cam_id):
    """Generate a synthetic 'CCTV frame' with overlay boxes/labels."""
    cfg = CAMERAS[cam_id]
    w, h = 900, 480
    img = Image.new("RGB", (w, h), cfg["bg"])
    draw = ImageDraw.Draw(img)

    try:
        # Use the exact filename of your downloaded font
        font_path = "custom_font.ttf" 
        ui_font = ImageFont.truetype(font_path, 16) 
        title_font = ImageFont.truetype(font_path, 20)
    except IOError:
        st.error(f"Critical Failure: Could not load {font_path}. Verify the file exists in the directory.")
        ui_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # faint grid to sell the "camera view" look
    for x in range(0, w, 45):
        draw.line([(x, 0), (x, h)], fill=tuple(min(c + 10, 255) for c in cfg["bg"]), width=1)
    for y in range(0, h, 45):
        draw.line([(0, y), (w, y)], fill=tuple(min(c + 10, 255) for c in cfg["bg"]), width=1)

    # simulated detection boxes
    random.seed(cam_id + str(st.session_state.tick // 3))
    worker_box = (
        random.randint(80, 300), random.randint(180, 280),
    )
    draw.rectangle(
        [worker_box[0], worker_box[1], worker_box[0] + 60, worker_box[1] + 140],
        outline=(80, 200, 255), width=2,
    )
    draw.text((worker_box[0], worker_box[1] - 18), "WORKER · person", fill=(80, 200, 255))

    machine_box = (
        random.randint(450, 650), random.randint(140, 220),
    )
    draw.rectangle(
        [machine_box[0], machine_box[1], machine_box[0] + 200, machine_box[1] + 160],
        outline=(255, 190, 60), width=2,
    )
    draw.text((machine_box[0], machine_box[1] - 18), "MACHINERY · active", fill=(255, 190, 60))

    violation = st.session_state.active_violations.get(cam_id)
    if violation:
        draw.rectangle([0, 0, w - 1, h - 1], outline=(239, 68, 68), width=6)
        draw.rectangle([20, h - 60, 20 + 9 * len(violation["violation"]), h - 24], fill=(60, 12, 14))
        draw.text((30, h - 52), f"⚠ {violation['violation']}", fill=(255, 120, 120))
    else:
        draw.rectangle([20, h - 60, 300, h - 24], fill=(10, 40, 24))
        draw.text((30, h - 52), "✓ Nominal — no active violation", fill=(120, 230, 160))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw.text((w - 230, 14), f"{cam_id} · LIVE · {ts}", fill=(200, 205, 220))
    draw.text((14, 14), cfg["name"], fill=(230, 232, 240))

    return img



st.session_state.tick += 1
for cid in CAMERAS:
    maybe_update_violation(cid)

# cap history length
st.session_state.alert_log = st.session_state.alert_log[:40]

sel = st.session_state.selected_cam
cfg = CAMERAS[sel]


top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown(
        f"<div class='gw-title'>🛰️ Sentinel SiteOps Vision</div>"
        f"<div class='gw-sub' style='margin-left: 65px;'>Gemma-4 edge reasoning agent · semantic safety monitoring, not raw detection</div>",
        unsafe_allow_html=True,
    )


st.write("")


col_cams, col_feed, col_alert = st.columns([0.8, 3.0, 1.2], gap="medium")

with col_cams:
    # 1. Removed the broken gw-card wrapper. Header is standalone.
    st.markdown("<div class='gw-header'>CCTV Cameras</div>", unsafe_allow_html=True)

    cam_ids = list(CAMERAS.keys())
    current_idx = cam_ids.index(st.session_state.selected_cam)

    selected_cid = st.selectbox(
        "Select Camera",
        options=cam_ids,
        index=current_idx,
        format_func=lambda x: f"{x} · {CAMERAS[x]['name']}",
        label_visibility="collapsed",
        key="cam_dropdown"
    )

    if selected_cid != st.session_state.selected_cam:
        st.session_state.selected_cam = selected_cid
        st.rerun()


with col_feed:
    # 2. Removed the broken gw-card wrapper. Header is standalone.
    st.markdown(
        f"<div class='gw-header'>Live Feed — {sel} · {cfg['zone']}</div>",
        unsafe_allow_html=True,
    )
    frame = render_frame(sel)
    st.image(frame, use_container_width=True)

    # 3. Consolidate pure HTML content into a single string execution. 
    # Because there are no Streamlit widgets inside this specific block, we can safely wrap it in the gw-card class.
    rules_html = f"<div class='gw-card'><div class='gw-header'>Zone Rules — {cfg['zone']}</div>"
    for r in cfg["rules"]:
        rules_html += f"<div class='gw-rule'>▸ {r}</div>"
    rules_html += "</div>"
    
    st.markdown(rules_html, unsafe_allow_html=True)

with col_alert:
    st.markdown("<div class='gw-header' style='margin-left:2px;'>Active Alert</div>", unsafe_allow_html=True)
    v = st.session_state.active_violations.get(sel)
    if v:
        st.markdown(
            f"""
            <div class='gw-alert-box'>
                <div class='gw-alert-title'>⚠ SAFETY VIOLATION — LEVEL A</div>
                <div class='gw-kv'><b>Violation:</b> {v['violation']}</div>
                <div class='gw-kv'><b>Worker ID:</b> {v['worker_id']}</div>
                <div class='gw-kv'><b>Zone:</b> {v['zone']}</div>
                <div class='gw-kv'><b>Duration:</b> {v['duration_s']}s</div>
                <div class='gw-kv'><b>Threshold:</b> {v['threshold_s']}s</div>
                <div class='gw-kv'><b>Detected:</b> {v['time']}</div>
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
    st.markdown(
        f"<div class='gw-card'><div class='gw-header'>Site Overview</div>"
        f"<div class='gw-kv'><b>Cameras online:</b> {len(CAMERAS)}</div>"
        f"<div class='gw-kv'><b>Active violations (site-wide):</b> {n_active}</div>"
        f"<div class='gw-kv'><b>Logged today:</b> {len(st.session_state.alert_log)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )



st.write("")
pad_l, table_col, pad_r = st.columns([0.4, 3.2, 0.4])
with table_col:
    st.markdown("<div class='gw-card'>", unsafe_allow_html=True)
    st.markdown("<div class='gw-header'>Recent Alerts</div>", unsafe_allow_html=True)

    rows = []
    for v in st.session_state.active_violations.values():
        rows.append(v)
    rows += st.session_state.alert_log

    if rows:
        table_data = [
            {
                "Time": r["time"],
                "Worker ID": r["worker_id"],
                "Zone": r["zone"],
                "Violation": r["violation"],
                "Duration": f"{r['duration_s']}s",
                "Status": r["status"],
            }
            for r in rows[:20]
        ]
        st.dataframe(table_data, use_container_width=True, hide_index=True)
    else:
        st.markdown("<div class='gw-sub'>No alerts logged yet this session.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.live:
    time.sleep(2.5)
    st.rerun()
