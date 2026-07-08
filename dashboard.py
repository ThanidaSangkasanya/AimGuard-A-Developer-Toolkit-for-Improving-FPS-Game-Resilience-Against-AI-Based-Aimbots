import streamlit as st
import subprocess
import os
import sys
import torch
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def sanitize_game_name(name: str) -> str:
    """Turns a free-text game name into a safe, consistent folder name so the
    same dataset/labels can be found again next time without re-uploading."""
    safe = (name or "").strip().lower().replace(" ", "_")
    return safe or "default_game"

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
.gt-hint {
    font-size: 12.5px;
    color: #64748B;
    margin-top: 6px;
}
.gt-hint a {
    color: #8B5CF6;
    font-weight: 600;
    text-decoration: none;
}
.gt-hint a:hover {
    text-decoration: underline;
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
if 'label_path' not in st.session_state:
    st.session_state.label_path = None
if 'custom_model_path' not in st.session_state:
    st.session_state.custom_model_path = None
if 'custom_model_name' not in st.session_state:
    st.session_state.custom_model_name = None
if 'show_add_model' not in st.session_state:
    st.session_state.show_add_model = False

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

        game = st.text_input(
            "Game name",
            value="cs2",
            placeholder="e.g. cs2, valorant, overwatch, your_game_name",
            help="Enter any game name — used to organize saved noise files, frames, and labels "
                 "so the same dataset can be reused later without re-uploading"
        )
        game_safe   = sanitize_game_name(game)
        frames_dir  = os.path.join("data", game_safe, "frames")
        labels_dir  = os.path.join("data", game_safe, "labels")

        # ── Check for an already-saved dataset for this game name ──
        existing_frames = []
        if os.path.exists(frames_dir):
            existing_frames = [f for f in os.listdir(frames_dir)
                               if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if existing_frames:
            st.success(
                f"📁 Found a saved dataset for **{game.strip() or game_safe}** → "
                f"**{len(existing_frames)}** frame(s) already in `{frames_dir}`. "
                f"No need to re-upload — you can go straight to Configuration below, "
                f"or upload more frames to add to this set."
            )

        uploaded_files = st.file_uploader(
            "Drop game frames here (JPEG / PNG)" if not existing_frames
            else "Add more frames (optional — merges into the saved set above)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Upload game screenshots to train the noise pattern. Frames are saved under "
                 "data/<game_name>/frames so you can reuse them next time just by typing the "
                 "same game name."
        )

        # Save uploaded files into the per-game frames folder (merges with existing)
        if uploaded_files:
            os.makedirs(frames_dir, exist_ok=True)
            for uf in uploaded_files:
                with open(os.path.join(frames_dir, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
            total_now = len([f for f in os.listdir(frames_dir)
                             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
            st.success(f"✓ {len(uploaded_files)} new frame(s) saved → total {total_now} frame(s) for '{game.strip() or game_safe}'")
            data_path = frames_dir
        elif existing_frames:
            data_path = frames_dir
        else:
            # fallback to .env DATASET_ROOT
            dataset_root = os.environ.get('DATASET_ROOT', '')
            data_path    = dataset_root if dataset_root else ""

        st.markdown("---")

        # ── Optional ground-truth label upload ─────────────
        existing_labels = []
        if os.path.exists(labels_dir):
            existing_labels = os.listdir(labels_dir)

        with st.expander(
            f"🏷️ Optional: Upload ground-truth labels"
            + (f"  ·  {len(existing_labels)} saved" if existing_labels else ""),
            expanded=False
        ):
            st.markdown(
                "Upload bounding-box labels for your dataset (e.g. YOLO-format `.txt`, "
                "or a single `.json` / `.csv` annotation file). These labels are optional "
                "for training the cloak, but they are used later on the **Evaluate** page "
                "to compute recall/precision-style metrics against ground truth.\n\n"
                "Labels are saved under `data/<game_name>/labels`, matched to this game name — "
                "so they're reused automatically next time too, no re-upload needed."
            )

            if existing_labels:
                st.info(f"📁 Found **{len(existing_labels)}** saved label file(s) for "
                       f"**{game.strip() or game_safe}** → `{labels_dir}`")

            uploaded_labels = st.file_uploader(
                "Add ground-truth label files" if existing_labels
                else "Drop ground-truth label files here",
                type=["txt", "json", "csv", "xml"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                help="One label file per image (YOLO .txt) or a single annotation file (.json/.csv/.xml)",
                key="gt_label_uploader"
            )

            if uploaded_labels:
                os.makedirs(labels_dir, exist_ok=True)
                for lf in uploaded_labels:
                    with open(os.path.join(labels_dir, lf.name), "wb") as f:
                        f.write(lf.getbuffer())
                st.success(f"✓ {len(uploaded_labels)} label file(s) saved → `{labels_dir}`")
                st.session_state.label_path = labels_dir
            elif existing_labels:
                st.session_state.label_path = labels_dir
            else:
                st.session_state.label_path = None

            st.markdown(
                "<div class='gt-hint'>Don't have ground-truth labels yet? "
                "<a href='https://www.makesense.ai/' target='_blank'>Click here</a> "
                "to annotate your frames for free with makesense.ai.</div>",
                unsafe_allow_html=True
            )

    # ── Configuration card (full width) ───────────────────
    with st.container(border=True):
        head_l, head_r = st.columns([5, 1])
        with head_l:
            st.markdown("#### ⚙️ Configuration")
        with head_r:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("➕ Add model", key="toggle_add_model", use_container_width=True):
                st.session_state.show_add_model = not st.session_state.show_add_model

        # ── Optional: add a custom detection model (top-right toggle) ──
        if st.session_state.show_add_model:
            with st.container(border=True):
                cm_l, cm_r = st.columns([5, 1])
                with cm_l:
                    st.markdown("**➕ Add a custom detection model**")
                with cm_r:
                    if st.button("✕ Close", key="close_add_model", use_container_width=True):
                        st.session_state.show_add_model = False
                        st.rerun()

                st.markdown(
                    "You can add your own cheating/detection model to train against, in addition "
                    "to the **3 built-in demo models** below (YOLOv5n, NanoDet-Plus, RT-DETR), "
                    "which represent common real-world visual-aimbot architectures.\n\n"
                    "**Supported formats:** PyTorch `.pt` weights only, compatible with the "
                    "**YOLO family** (Ultralytics-style YOLOv5 / YOLOv8 export). Other "
                    "architectures (e.g. NanoDet, RT-DETR, Detectron2, TensorFlow, ONNX) are "
                    "not supported for custom upload — use one of the 3 built-in demo models instead."
                )

                custom_model_file = st.file_uploader(
                    "Upload a YOLO-compatible .pt weight file",
                    type=["pt"],
                    accept_multiple_files=False,
                    key="custom_model_uploader"
                )
                custom_model_display_name = st.text_input(
                    "Display name for this model",
                    value="custom_yolo",
                    key="custom_model_name_input",
                    help="Used to label this model in the proxy-model selector and in saved noise paths"
                )

                custom_dir = os.path.join("pretrained_models", "custom")
                if custom_model_file is not None:
                    os.makedirs(custom_dir, exist_ok=True)
                    safe_name  = custom_model_display_name.strip().lower().replace(" ", "_") or "custom_yolo"
                    saved_path = os.path.join(custom_dir, f"{safe_name}.pt")
                    with open(saved_path, "wb") as f:
                        f.write(custom_model_file.getbuffer())
                    st.session_state.custom_model_path = saved_path
                    st.session_state.custom_model_name = safe_name
                    st.success(f"✓ Custom model saved → {saved_path}")
                elif st.session_state.custom_model_path and os.path.exists(st.session_state.custom_model_path):
                    st.info(f"Using previously uploaded model → `{st.session_state.custom_model_path}`")

            st.markdown("---")

        col_a, col_b = st.columns(2)
        with col_a:
            model_options = ["yolov5n", "nanodet", "rtdetr"]
            model_labels  = {
                "yolov5n": "YOLOv5n (demo)",
                "nanodet": "NanoDet-Plus (demo)",
                "rtdetr":  "RT-DETR ⚠️ slow (demo)"
            }
            if st.session_state.custom_model_path and os.path.exists(st.session_state.custom_model_path):
                model_options.append("custom")
                model_labels["custom"] = f"⭐ {st.session_state.custom_model_name} (custom)"

            proxy_model = st.radio(
                "Aimbot model (1 noise file per model)",
                model_options,
                format_func=lambda x: model_labels[x],
                help="Each model generates one separate noise file. The 3 built-in models "
                     "are demo/reference cheating architectures; the custom option (if "
                     "added via 'Add model' above) trains against your own YOLO-compatible weights."
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
            ssim_w = st.slider(
                "SSIM weight",
                min_value=0.0, max_value=1.0, value=0.3, step=0.05,
                help="Controls the trade-off between attack strength and image quality. "
                     "Higher = noise is pushed to preserve visual similarity to the original "
                     "frame (safer/less visible, but DSR may drop). Lower = the optimizer "
                     "focuses more on driving detection confidence to 0 (higher DSR, but the "
                     "cloak may become more visible). Default: 0.3"
            )
            st.caption(
                "💡 If DSR is too low, try lowering SSIM weight first. "
                "If the cloak looks too visible in the preview, raise it back up."
            )

    # ── Aimbot Vision Preview (before Invisibility Cloak) ──
    with st.container(border=True):
        st.markdown("#### 🎯 Aimbot Vision Preview")
        st.markdown(
            "<div class='gt-hint'>See exactly what a visual aimbot detects on your "
            "<b>raw, unprotected</b> frames — before any Invisibility Cloak noise is applied. "
            "Useful as a before/after reference once you've trained a cloak.</div>",
            unsafe_allow_html=True
        )

        preview_upload_dir = frames_dir
        preview_images = []
        if os.path.exists(preview_upload_dir):
            preview_images = sorted([
                f for f in os.listdir(preview_upload_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])

        if not preview_images:
            st.info("Upload game frames in the Dataset section above to enable the preview.")
        else:
            pv_c1, pv_c2 = st.columns([2, 1])
            with pv_c1:
                preview_image_name = st.selectbox(
                    "Choose a frame to inspect",
                    preview_images,
                    key="aimbot_preview_image"
                )
            with pv_c2:
                preview_model_options = ["yolov5n", "nanodet", "rtdetr"]
                preview_model_labels  = {
                    "yolov5n": "YOLOv5n",
                    "nanodet": "NanoDet-Plus",
                    "rtdetr":  "RT-DETR"
                }
                if st.session_state.custom_model_path and os.path.exists(st.session_state.custom_model_path):
                    preview_model_options.append("custom")
                    preview_model_labels["custom"] = f"⭐ {st.session_state.custom_model_name}"
                preview_model = st.selectbox(
                    "Model",
                    preview_model_options,
                    key="aimbot_preview_model",
                    format_func=lambda x: preview_model_labels[x]
                )

            preview_conf = st.slider("Detection confidence threshold", 0.1, 0.9, 0.4, 0.05,
                                     key="aimbot_preview_conf")

            if st.button("▶️ Run Aimbot Preview", use_container_width=True):
                img_path = os.path.join(preview_upload_dir, preview_image_name)
                out_dir  = os.path.join("result", "preview")
                os.makedirs(out_dir, exist_ok=True)

                cmd = [
                    sys.executable, "preview_detect.py",
                    "--image", img_path,
                    "--model", preview_model,
                    "--conf",  str(preview_conf),
                    "--out_dir", out_dir,
                ]
                if preview_model == "custom":
                    cmd += ["--custom_model_path", st.session_state.custom_model_path]

                with st.spinner("Running detection on the raw frame ..."):
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True,
                        encoding="utf-8", errors="replace",
                        cwd=os.path.abspath(os.path.dirname(__file__)) or os.getcwd()
                    )

                base_name = os.path.splitext(preview_image_name)[0]
                out_img   = os.path.join(out_dir, f"{base_name}_aimbot_view.jpg")

                if os.path.exists(out_img):
                    pv_res1, pv_res2 = st.columns(2)
                    with pv_res1:
                        st.markdown("**Original frame**")
                        st.image(img_path, use_container_width=True)
                    with pv_res2:
                        st.markdown("**Before Invisibility Cloak — aimbot sees this**")
                        st.image(out_img, use_container_width=True)
                    st.caption(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "")
                else:
                    st.error("Preview failed to generate an output image.")
                    with st.expander("Error log"):
                        st.text(proc.stdout + "\n" + proc.stderr)

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
        st.info(f"Ready to train **{proxy_model}** | epsilon={epsilon}/255 | ssim_w={ssim_w} | epochs={n_iter} | game={game}")

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
                        "--game",        game_safe,
                        "--model",       model_name,
                        "--n_iter",      str(n_iter),
                        "--lr",          str(lr),
                        "--epsilon",     str(epsilon),
                        "--ssim_w",      str(ssim_w),
                        "--data_path",   data_path,
                    ]

                    with st.spinner(f"Training {model_name} ..."):
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
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
                            st.session_state.last_game  = game_safe
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

    st.markdown(
        "<div class='gt-hint'>Datasets and labels uploaded on the <b>Train noise</b> page are "
        "saved per game name and reused here automatically — no re-upload needed. Don't have "
        "ground-truth labels yet? "
        "<a href='https://www.makesense.ai/' target='_blank'>Click here</a> to annotate frames.</div>",
        unsafe_allow_html=True
    )
    st.markdown("")

    # Summary metrics — always show cards, 0 if no data
    eval_csv   = os.path.join("result", "evaluation_summary.csv")
    df_summary = None
    latest     = None
    if os.path.exists(eval_csv):
        df_summary = pd.read_csv(eval_csv)
        if len(df_summary) > 0:
            latest = df_summary.iloc[-1]

    dsr_val    = latest["DSR"] * 100   if latest is not None else 0.0
    frames_val = int(latest["Images"]) if latest is not None else 0
    sub_val    = f"{latest['Game']} / {latest['Model']}" if latest is not None else "No data yet"

    has_gt_metrics = latest is not None and str(latest.get("Recall_After", "")).strip() not in ("", "nan")
    recall_before_val = float(latest["Recall_Before"]) * 100 if has_gt_metrics else None
    recall_after_val  = float(latest["Recall_After"])  * 100 if has_gt_metrics else None
    recall_drop_val   = float(latest["Recall_Drop"])   * 100 if has_gt_metrics else None
    gt_images_val = int(latest["GT_Images"]) if latest is not None and str(latest.get("GT_Images", "")).strip() not in ("", "nan") else 0

    cols = st.columns(5) if has_gt_metrics else st.columns(2)
    c1, c2 = cols[0], cols[1]
    dsr_color = "#15803D" if dsr_val > 0 else "#94A3B8"
    num_color = "#0F172A" if frames_val > 0 else "#94A3B8"
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Overall DSR</div>
            <div class='metric-val' style='color:{dsr_color}'>{dsr_val:.1f}%</div>
            <div class='metric-sub'>{sub_val}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Frames Tested</div>
            <div class='metric-val' style='color:{num_color}'>{frames_val}</div>
            <div class='metric-sub'>{sub_val}</div>
        </div>""", unsafe_allow_html=True)
    if has_gt_metrics:
        c3, c4, c5 = cols[2], cols[3], cols[4]
        with c3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Recall Before cloak</div>
                <div class='metric-val' style='color:#B91C1C'>{recall_before_val:.1f}%</div>
                <div class='metric-sub'>{gt_images_val} labeled frames</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Recall After cloak</div>
                <div class='metric-val' style='color:#B45309'>{recall_after_val:.1f}%</div>
                <div class='metric-sub'>{gt_images_val} labeled frames</div>
            </div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Recall Drop</div>
                <div class='metric-val' style='color:#15803D'>{recall_drop_val:.1f} pts</div>
                <div class='metric-sub'>higher = stronger defense</div>
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
            _model_list   = ["yolov5n", "nanodet", "rtdetr"]
            _model_labels = {
                "yolov5n": "YOLOv5n",
                "nanodet": "NanoDet-Plus",
                "rtdetr":  "RT-DETR"
            }
            if st.session_state.custom_model_path and os.path.exists(st.session_state.custom_model_path):
                _model_list.append("custom")
                _model_labels["custom"] = f"⭐ {st.session_state.custom_model_name} (custom)"
            _model_default = st.session_state.get('last_model', 'yolov5n')
            _model_idx   = _model_list.index(_model_default) if _model_default in _model_list else 0
            eval_model = st.selectbox("Model", _model_list,
                                      index=_model_idx,
                                      key="eval_model",
                                      format_func=lambda x: _model_labels[x])
        with ec2:
            eval_conf    = st.slider("Confidence threshold (DSR)", 0.1, 0.9, 0.4, 0.05)
            eval_epsilon = st.select_slider("Perturbation bound",
                                            options=[4, 8, 16, 32], value=8,
                                            format_func=lambda x: f"{x}/255",
                                            key="eval_eps")

        # ── Auto-detect the same per-game frames/labels saved on Train noise ──
        eval_game_safe  = sanitize_game_name(eval_game)
        eval_frames_dir = os.path.join("data", eval_game_safe, "frames")
        eval_labels_dir = os.path.join("data", eval_game_safe, "labels")

        eval_frame_count = 0
        if os.path.exists(eval_frames_dir):
            eval_frame_count = len([f for f in os.listdir(eval_frames_dir)
                                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        if eval_frame_count > 0:
            st.caption(f"📁 Using saved dataset for '{eval_game.strip() or eval_game_safe}' → "
                      f"{eval_frame_count} frame(s) from `{eval_frames_dir}` (no re-upload needed)")
        else:
            st.warning(f"⚠️ No saved frames found for '{eval_game.strip() or eval_game_safe}'. "
                      f"Go to **Train noise** and upload frames under this game name first.")

        eval_labels_exist = os.path.exists(eval_labels_dir) and os.listdir(eval_labels_dir)

        use_gt  = False
        iou_thr = 0.5
        if eval_labels_exist:
            gt_c1, gt_c2 = st.columns([2, 1])
            with gt_c1:
                use_gt = st.checkbox(
                    f"Use saved ground-truth labels ({len(os.listdir(eval_labels_dir))} files) for this evaluation",
                    value=True,
                    help="If checked, get_cloak.py will be passed --label_path to compute "
                         "Recall Before/After/Drop against your ground truth. "
                         "Labels must be YOLO-format .txt files with the same basename as "
                         "the image."
                )
            with gt_c2:
                iou_thr = st.slider("IoU match threshold", 0.1, 0.9, 0.5, 0.05,
                                    disabled=not use_gt)

        save_gallery = st.checkbox(
            "Save before/after comparison images (optional)",
            value=True,
            help="Saves raw-vs-cloaked frames, both with and without bounding boxes, "
                 "so you can visually compare them below after evaluation finishes."
        )

        noise_path = os.path.join("universal_cloak",
                                  eval_game_safe,
                                  eval_model, "universal_noise.pt")
        if os.path.exists(noise_path):
            st.success(f"✓ Noise file found: {noise_path}")
        else:
            st.warning(f"⚠️ Noise not found: {noise_path} — run Train noise first")

        if st.button("▶️ Run Evaluation", use_container_width=True, type="primary"):
            if not os.path.exists(noise_path):
                st.error("Noise file not found. Train first.")
            elif eval_frame_count == 0:
                st.error("No frames found for this game. Upload frames on the Train noise page first.")
            else:
                cmd = [
                    sys.executable, "get_cloak.py",
                    "--game",         eval_game_safe,
                    "--model",        eval_model,
                    "--conf",         str(eval_conf),
                    "--epsilon",      str(eval_epsilon),
                    "--save_gallery", str(save_gallery),
                ]
                if eval_model == "custom" and st.session_state.custom_model_path:
                    cmd += ["--custom_model_path", st.session_state.custom_model_path]
                # reuse the same per-game frames folder saved on Train noise
                if eval_frame_count > 0:
                    cmd += ["--data_path", eval_frames_dir]
                # pass ground-truth label path if the user opted in
                if use_gt and eval_labels_exist:
                    cmd += ["--label_path", eval_labels_dir,
                            "--iou_thr", str(iou_thr)]
            with st.spinner("Evaluating (before vs after cloak) ..."):
                _env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
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
            st.markdown("#### DSR & Recall Before/After Chart")
            st.image(chart_path, use_column_width=True)

    # ── Optional: Before / After image comparison gallery ──
    if 'eval_game' in dir() and 'eval_model' in dir():
        _g = sanitize_game_name(eval_game)
        _eval_root      = os.path.join("result", "evaluation", _g, eval_model)
        _before_clean   = os.path.join(_eval_root, "before_clean")
        _before_bbox    = os.path.join(_eval_root, "before_bbox")
        _after_clean    = os.path.join(_eval_root, "after_clean")
        _after_bbox     = os.path.join(_eval_root, "after_bbox")

        if os.path.exists(_after_clean) and os.listdir(_after_clean):
            with st.container(border=True):
                st.markdown("#### 🖼️ Optional: Before / After Comparison")
                st.markdown(
                    "<div class='gt-hint'>Compare a frame before and after Invisibility Cloak "
                    "was applied, with or without bounding boxes, to see how visible the noise "
                    "is and how much the aimbot's detections change. Red = predicted box, "
                    "Green = ground truth (only shown when labels were used during evaluation).</div>",
                    unsafe_allow_html=True
                )

                # base filenames available (from the 'after_clean' folder)
                _bases = sorted([
                    f[:-len("_after.jpg")] for f in os.listdir(_after_clean)
                    if f.endswith("_after.jpg")
                ])

                if _bases:
                    gal_c1, gal_c2 = st.columns([2, 1])
                    with gal_c1:
                        chosen_base = st.selectbox("Choose a frame", _bases, key="gallery_frame")
                    with gal_c2:
                        show_bbox = st.checkbox("Show bounding boxes", value=True, key="gallery_bbox")

                    if show_bbox:
                        before_path = os.path.join(_before_bbox, f"{chosen_base}_before_bbox.jpg")
                        after_path  = os.path.join(_after_bbox,  f"{chosen_base}_after_bbox.jpg")
                    else:
                        before_path = os.path.join(_before_clean, f"{chosen_base}_before.jpg")
                        after_path  = os.path.join(_after_clean,  f"{chosen_base}_after.jpg")

                    img_c1, img_c2 = st.columns(2)
                    with img_c1:
                        st.markdown("**Before — raw frame (aimbot's original view)**")
                        if os.path.exists(before_path):
                            st.image(before_path, use_container_width=True)
                        else:
                            st.info("Image not found — rerun evaluation with the gallery option enabled.")
                    with img_c2:
                        st.markdown("**After — with Invisibility Cloak**")
                        if os.path.exists(after_path):
                            st.image(after_path, use_container_width=True)
                        else:
                            st.info("Image not found — rerun evaluation with the gallery option enabled.")
                else:
                    st.info("No gallery images found for this game/model yet.")


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