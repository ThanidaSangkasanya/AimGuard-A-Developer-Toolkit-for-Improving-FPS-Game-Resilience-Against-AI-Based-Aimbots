import streamlit as st
import subprocess
import os
import sys
import torch
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="AimGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stDeployButton"] {display: none;}
[data-testid="stSidebar"] {
    background-color: #0F172A;
}
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
[data-testid="stSidebarNav"] {
    display: none;
}
.logo-text {
    font-size: 22px;
    font-weight: 700;
    color: #F1F5F9 !important;
    letter-spacing: 0.05em;
}
.logo-sub {
    font-size: 13px;
    color: #64748B !important;
}
.nav-active {
    background: rgba(139, 92, 246, 0.15);
    border-left: 3px solid #8B5CF6;
    border-radius: 6px;
    padding: 8px 12px;
    color: #A78BFA !important;
    font-weight: 600;
}
.page-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-color, inherit);
}
.page-sub {
    font-size: 14px;
    color: #8B5CF6;
    margin-top: -8px;
    margin-bottom: 20px;
}
.metric-card {
    background: var(--background-color, #F8FAFC);
    border: 1px solid var(--secondary-background-color, #E2E8F0);
    border-radius: 12px;
    padding: 20px;
    text-align: left;
}
.metric-label {
    font-size: 13px;
    color: var(--text-color, #64748B);
    opacity: 0.7;
    margin-bottom: 6px;
}
.metric-val {
    font-size: 28px;
    font-weight: 700;
}
.metric-sub {
    font-size: 11px;
    color: var(--text-color, #94A3B8);
    opacity: 0.5;
    margin-top: 4px;
}
.status-dot-green { color: #22C55E; font-size: 12px; }
.status-dot-amber { color: #F59E0B; font-size: 12px; }
.code-block {
    background: #1E293B;
    border-radius: 8px;
    padding: 16px;
    font-family: monospace;
    font-size: 13px;
    color: #E2E8F0;
    line-height: 1.8;
}
.section-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'train'
if 'training_done' not in st.session_state:
    st.session_state.training_done = False
if 'eval_done' not in st.session_state:
    st.session_state.eval_done = False

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; padding:8px 0 20px;
                border-bottom:1px solid #1E293B; margin-bottom:16px'>
        <div style='width:36px;height:36px;background:#1E3A5F;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;font-size:18px'>🛡️</div>
        <div>
            <div class='logo-text'>AIMGUARD</div>
            <div class='logo-sub'>Developer toolkit</div>
        </div>
    </div>
    <div style='font-size:11px;color:#475569;letter-spacing:0.08em;
                padding:0 4px 8px'>TOOLKIT</div>
    """, unsafe_allow_html=True)

    if st.button("⚙️  Train noise",  use_container_width=True,
                 type="primary" if st.session_state.page == 'train' else "secondary"):
        st.session_state.page = 'train'
        st.rerun()

    if st.button("📊  Evaluate",     use_container_width=True,
                 type="primary" if st.session_state.page == 'eval' else "secondary"):
        st.session_state.page = 'eval'
        st.rerun()

    if st.button("🔗  Integrate",    use_container_width=True,
                 type="primary" if st.session_state.page == 'integrate' else "secondary"):
        st.session_state.page = 'integrate'
        st.rerun()

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    # GPU status
    gpu_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "Not available"
    st.markdown(f"""
    <div style='border-top:1px solid #1E293B; padding-top:16px; font-size:12px'>
        <div style='color:#{"22C55E" if gpu_ok else "EF4444"}'>
            {"● " + gpu_name if gpu_ok else "● GPU not found"}
        </div>
        <div style='color:#475569; margin-top:4px'>
            {"CUDA " + torch.version.cuda if gpu_ok and torch.version.cuda else "CPU mode"}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE: TRAIN NOISE
# ════════════════════════════════════════════════════════════
if st.session_state.page == 'train':
    st.markdown("<div class='page-title'>Train noise</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Generate a Universal Noise pattern from game dataset</div>",
                unsafe_allow_html=True)

    # ── Dataset card (full width) ──────────────────────────
    with st.container(border=True):
        st.markdown("#### 📁 Dataset")

        uploaded_files = st.file_uploader(
            "Drop game frames here (JPEG / PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Upload game screenshots to train the noise pattern"
        )

        # Save uploaded files to temp folder
        upload_dir = os.path.join("data", "uploaded_frames")
        if uploaded_files:
            os.makedirs(upload_dir, exist_ok=True)
            for uf in uploaded_files:
                with open(os.path.join(upload_dir, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
            st.success(f"✓ {len(uploaded_files)} game frames uploaded")
            data_path = upload_dir
        else:
            # fallback to .env DATASET_ROOT
            dataset_root = os.environ.get('DATASET_ROOT', '')
            data_path    = dataset_root if dataset_root else ""

        game = st.text_input(
            "Game name",
            value="cs2",
            placeholder="e.g. cs2, valorant, overwatch, your_game_name",
            help="Enter any game name — used to organize saved noise files"
        )

    # ── Configuration card (full width) ───────────────────
    with st.container(border=True):
        st.markdown("#### ⚙️ Configuration")

        col_a, col_b = st.columns(2)
        with col_a:
            proxy_model = st.radio(
                "Proxy model (1 noise file per model)",
                ["yolov5n", "nanodet", "rtdetr"],
                format_func=lambda x: {
                    "yolov5n": "YOLOv5n",
                    "nanodet": "NanoDet-Plus",
                    "rtdetr":  "RT-DETR ⚠️ slow"
                }[x],
                help="Each model generates one separate noise file"
            )
            proxy_models = [proxy_model]
            epsilon = st.select_slider(
                "Perturbation bound",
                options=[4, 8, 16, 32],
                value=8,
                format_func=lambda x: f"{x}/255"
            )
        with col_b:
            n_iter = st.number_input("Training epochs", min_value=10,
                                     max_value=500, value=100, step=10)
            lr     = st.number_input("Learning rate", min_value=0.0001,
                                     max_value=0.01, value=0.0005, step=0.0001,
                                     format="%.4f")

    # ── Warning banner ─────────────────────────────────────
    st.info(
        "ℹ️ **Training time depends on:** number of frames, training epochs, "
        "GPU performance, and selected model.\n\n"
        "Each model produces **1 separate noise file** "
        f"→ `universal_cloak/{game.strip() or 'game'}/{proxy_model}/universal_noise.pt`"
    )
    if proxy_model == "rtdetr":
        st.warning(
            "⚠️ **RT-DETR selected** — this model uses a Transformer architecture "
            "which requires significantly more memory and compute time than YOLOv5n or NanoDet-Plus. "
            "Training may take several times longer. Ensure you have a capable GPU before proceeding."
        )

    st.markdown("---")

    if not game.strip():
        st.error("Please enter a game name.")
    else:
        st.info(f"Ready to train **{proxy_model}** | epsilon={epsilon}/255 | epochs={n_iter} | game={game}")

        if st.button("🚀 Start Training", use_container_width=True, type="primary"):
            if not data_path or not os.path.exists(data_path):
                st.error("No frames found. Please upload game frames above.")
            else:
                for model_name in proxy_models:
                    st.markdown(f"**Training: {model_name.upper()}**")
                    progress_bar = st.progress(0)
                    status_text  = st.empty()

                    cmd = [
                        sys.executable, "train_cloak.py",
                        "--game",        game.strip().lower().replace(" ", "_"),
                        "--model",       model_name,
                        "--n_iter",      str(n_iter),
                        "--lr",          str(lr),
                        "--epsilon",     str(epsilon),
                        "--data_path",   data_path,
                    ]

                    with st.spinner(f"Training {model_name} ..."):
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            cwd=os.path.abspath(os.path.dirname(__file__)) or os.getcwd()
                        )
                        log_output = []
                        log_box    = st.expander("Training log", expanded=True)

                        for line in process.stdout:
                            line = line.rstrip()
                            log_output.append(line)
                            with log_box:
                                st.text(line)
                            # update progress from log
                            if "/" in line and "iter" not in line.lower():
                                try:
                                    parts = [p for p in line.split() if "/" in p]
                                    for part in parts:
                                        cur, total = part.split("/")
                                        pct = int(cur) / int(total)
                                        progress_bar.progress(pct)
                                        status_text.text(f"Epoch {cur}/{total}")
                                except Exception:
                                    pass

                        process.wait()
                        noise_path = os.path.join(
                            "universal_cloak", game, model_name, "universal_noise.pt")
                        # Success if noise file was created (returncode may be nonzero
                        # from harmless matplotlib/backend warnings)
                        if os.path.exists(noise_path):
                            progress_bar.progress(1.0)
                            status_text.text("Complete!")
                            st.success(f"✓ Noise saved → {noise_path}")
                            st.session_state.training_done = True
                            st.session_state.last_game  = game.strip().lower().replace(" ", "_")
                            st.session_state.last_model = model_name
                        else:
                            st.error(f"Training failed for {model_name} (exit code {process.returncode})")

        # Show existing noise files
        st.markdown("#### Trained noise files")
        noise_dir = os.path.join("universal_cloak", game)
        if os.path.exists(noise_dir):
            found = []
            for model_dir in os.listdir(noise_dir):
                pt_path = os.path.join(noise_dir, model_dir, "universal_noise.pt")
                if os.path.exists(pt_path):
                    size_mb = os.path.getsize(pt_path) / 1024 / 1024
                    found.append({"Model": model_dir, "File": pt_path,
                                  "Size": f"{size_mb:.2f} MB"})
            if found:
                st.dataframe(pd.DataFrame(found), use_container_width=True)
            else:
                st.info("No trained noise files yet for this game.")
        else:
            st.info("No trained noise files yet.")


# ════════════════════════════════════════════════════════════
# PAGE: EVALUATE
# ════════════════════════════════════════════════════════════
elif st.session_state.page == 'eval':
    st.markdown("<div class='page-title'>Evaluate</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Test how well the noise protects against aimbots</div>",
                unsafe_allow_html=True)

    # Summary metrics — always show cards, 0 if no data
    eval_csv   = os.path.join("result", "evaluation_summary.csv")
    df_summary = None
    latest     = None
    if os.path.exists(eval_csv):
        df_summary = pd.read_csv(eval_csv)
        if len(df_summary) > 0:
            latest = df_summary.iloc[-1]

    dsr_val    = latest["DSR"] * 100   if latest is not None else 0.0
    fps_val    = latest["AvgFPS"]      if latest is not None else 0.0
    frames_val = int(latest["Images"]) if latest is not None else 0
    sub_val    = f"{latest['Game']} / {latest['Model']}" if latest is not None else "No data yet"

    c1, c2, c3 = st.columns(3)
    dsr_color = "#15803D" if dsr_val > 0 else "#94A3B8"
    num_color = "#0F172A" if fps_val > 0 else "#94A3B8"
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Overall DSR</div>
            <div class='metric-val' style='color:{dsr_color}'>{dsr_val:.1f}%</div>
            <div class='metric-sub'>3 architectures</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Speed</div>
            <div class='metric-val' style='color:{num_color}'>{fps_val:.0f} FPS</div>
            <div class='metric-sub'>avg per frame</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Frames Tested</div>
            <div class='metric-val' style='color:{num_color}'>{frames_val}</div>
            <div class='metric-sub'>{sub_val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # DSR bars — always show, 0 if no data
    with st.container(border=True):
        st.markdown("#### DSR by architecture")
        if df_summary is not None and len(df_summary) > 0:
            for _, row in df_summary.iterrows():
                label = f"{row['Game']} / {row['Model']}"
                dsr   = float(row['DSR'])
                st.markdown(f"**{label}**")
                st.progress(dsr, text=f"{dsr*100:.1f}%")
        else:
            for arch in ["YOLOv5n", "NanoDet-Plus", "RT-DETR"]:
                st.markdown(f"**{arch}**")
                st.progress(0.0, text="0.0%")

    with st.container(border=True):
        st.markdown("#### ⚙️ Run Evaluation")

        ec1, ec2 = st.columns(2)
        with ec1:
            eval_game = st.text_input(
                "Game name",
                value=st.session_state.get('last_game', 'cs2'),
                key="eval_game",
                placeholder="e.g. cs2, valorant, your_game_name"
            )
            _model_list  = ["yolov5n", "nanodet", "rtdetr"]
            _model_default = st.session_state.get('last_model', 'yolov5n')
            _model_idx   = _model_list.index(_model_default) if _model_default in _model_list else 0
            eval_model = st.selectbox("Model", _model_list,
                                      index=_model_idx,
                                      key="eval_model",
                                      format_func=lambda x: {
                                          "yolov5n": "YOLOv5n",
                                          "nanodet": "NanoDet-Plus",
                                          "rtdetr":  "RT-DETR"
                                      }[x])
        with ec2:
            eval_conf    = st.slider("Confidence threshold (DSR)", 0.1, 0.9, 0.4, 0.05)
            eval_epsilon = st.select_slider("Perturbation bound",
                                            options=[4, 8, 16, 32], value=8,
                                            format_func=lambda x: f"{x}/255",
                                            key="eval_eps")

        noise_path = os.path.join("universal_cloak",
                                  eval_game.strip().lower().replace(" ", "_"),
                                  eval_model, "universal_noise.pt")
        if os.path.exists(noise_path):
            st.success(f"✓ Noise file found: {noise_path}")
        else:
            st.warning(f"⚠️ Noise not found: {noise_path} — run Train noise first")

        if st.button("▶️ Run Evaluation", use_container_width=True, type="primary"):
            if not os.path.exists(noise_path):
                st.error("Noise file not found. Train first.")
            else:
                cmd = [
                    sys.executable, "get_cloak.py",
                    "--game",    eval_game.strip().lower().replace(" ", "_"),
                    "--model",   eval_model,
                    "--conf",    str(eval_conf),
                    "--epsilon", str(eval_epsilon),
                ]
                # use uploaded frames if available
                _upload_dir = os.path.join("data", "uploaded_frames")
                if os.path.exists(_upload_dir) and os.listdir(_upload_dir):
                    cmd += ["--data_path", _upload_dir]
            with st.spinner("Evaluating ..."):
                _env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=_env,
                    cwd=os.path.abspath(os.path.dirname(__file__)) or os.getcwd()
                )
                _eval_log = []
                _log_area = st.empty()
                for line in process.stdout:
                    _eval_log.append(line.rstrip())
                    # show only last 15 lines, update in place
                    _log_area.text("\n".join(_eval_log[-15:]))
                process.wait()
                _summary = os.path.join("result", "evaluation_summary.csv")
                if os.path.exists(_summary):
                    st.success("✓ Evaluation complete!")
                    st.session_state.eval_done = True
                    st.rerun()
                else:
                    st.error(f"Evaluation failed (exit code {process.returncode}).")

    st.markdown("</div>", unsafe_allow_html=True)

    # Show per-image log if exists
    log_path = os.path.join("result", "log", eval_game,
                            f"{eval_game}_{eval_model}_eval.csv") \
        if 'eval_game' in dir() else None
    if log_path and os.path.exists(log_path):
        st.markdown("#### Per-frame log")
        df_log = pd.read_csv(log_path)
        st.dataframe(df_log.tail(20), use_container_width=True)

        chart_path = os.path.join("result", "evaluation_summary.png")
        if os.path.exists(chart_path):
            st.markdown("#### DSR & Recall Drop Chart")
            st.image(chart_path, use_column_width=True)


# ════════════════════════════════════════════════════════════
# PAGE: INTEGRATE
# ════════════════════════════════════════════════════════════
elif st.session_state.page == 'integrate':
    st.markdown("<div class='page-title'>Integrate</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Add AimGuard to your game rendering pipeline</div>",
                unsafe_allow_html=True)

    # Status
    with st.container(border=True):
     st.markdown("#### ✅ Status")

    gpu_ok   = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else None

    st.markdown(f"{'🟢' if gpu_ok else '🔴'} **GPU mode** — "
                f"{'CUDA ' + torch.version.cuda + ' · ' + gpu_name if gpu_ok else 'Not available'}")

    noise_files = []
    if os.path.exists("universal_cloak"):
        for game_dir in os.listdir("universal_cloak"):
            for model_dir in os.listdir(os.path.join("universal_cloak", game_dir)):
                pt = os.path.join("universal_cloak", game_dir, model_dir, "universal_noise.pt")
                if os.path.exists(pt):
                    noise_files.append(f"{game_dir}/{model_dir}/universal_noise.pt")

    if noise_files:
        st.markdown(f"🟢 **Noise files loaded** — {len(noise_files)} file(s) available")
        for nf in noise_files:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{nf}`")
    else:
        st.markdown("🔴 **No noise files** — run Train noise first")

    train_status = "🟡 Training in progress" \
        if st.session_state.training_done else "⚪ No active training"
    st.markdown(train_status)
    st.markdown("</div>", unsafe_allow_html=True)

    # Quick start
    with st.container(border=True):
     st.markdown("#### 💻 Quick Start")

    st.markdown("**Install**")
    st.code("pip install aimguard", language="bash")

    st.markdown("**Usage**")
    st.code("""# Load the trained noise file
from aimguard import NoiseEngine

engine    = NoiseEngine("universal_cloak/cs2/yolov5n/universal_noise.pt")
protected = engine.generate_noise(frame)

# frame    = numpy array (H x W x 3, uint8) from your game renderer
# protected = same shape — pass this to your display output""",
            language="python")
    st.markdown("</div>", unsafe_allow_html=True)

    # Integration guide
    with st.container(border=True):
     st.markdown("#### 🎮 Integration points by setup")
    st.markdown("""
| Setup | Where to hook in |
|---|---|
| Custom game engine | After render pass, before buffer swap |
| Unity | `OnRenderImage()` post-processing callback |
| Unreal Engine | Custom PostProcess Material or SceneCapture |
| Cloud Gaming server | Server-side before video encoding |
""")
    st.markdown("</div>", unsafe_allow_html=True)

    # Download noise file
    if noise_files:
        st.markdown("#### 📥 Download noise file")
        selected = st.selectbox("Select noise file", noise_files)
        with open(selected, "rb") as f:
            st.download_button(
                label="⬇️ Download .pt file",
                data=f,
                file_name=os.path.basename(selected),
                mime="application/octet-stream"
            )
    st.markdown("</div>", unsafe_allow_html=True)