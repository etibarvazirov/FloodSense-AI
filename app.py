"""FloodSense AI v3 — research dashboard. Educational prototype only."""
import os
import io
import csv
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="FloodSense AI | Research Dashboard", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
.stApp {background: #081322; color: #eaf1fb;}
.block-container {max-width: 1190px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 28px 32px; border: 1px solid #25415b; border-radius: 20px;
 background: linear-gradient(115deg,#112c42 0%,#10253a 55%,#0a1c2c 100%); margin-bottom: 18px;}
.eyebrow {color:#67d9ce; font-size:12px; font-weight:750; letter-spacing:2px;}
.hero h1 {font-size: 43px; margin: 9px 0 4px; color:#f6fbff;}
.hero p {color:#c4d4e3; font-size:16px; margin: 0;}
.smallcard {background:#10263a; border:1px solid #27435a; border-radius:16px; padding:17px 19px; min-height:114px;}
.kicker {color:#a3b9c9; font-size:12px; letter-spacing:1px; text-transform:uppercase;}
.bigresult {font-size:43px; color:#72e2d3; font-weight:800; margin:7px 0;}
.muted {color:#adbfcd; font-size:14px;}
.section-title {font-size:19px; font-weight:750; margin:6px 0 12px;}
.notice {background:#13283a; border-left:3px solid #eab96b; padding:14px 18px; border-radius:7px; color:#e3eaf0;}
.level {background:#11263a; border:1px solid #274159; padding:15px; border-radius:12px; min-height:106px;}
.level strong {color:#8be8db;}
footer {visibility:hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LEVELS = [f"Level{i}" for i in range(5)]
DESCRIPTIONS = [
    "Lowest visible flood-severity category in the dataset.",
    "Low flood-severity category.",
    "Intermediate flood-severity category.",
    "High flood-severity category.",
    "Highest visible flood-severity category in the dataset.",
]
REPO = "etibarvazirov/FloodSense-AI-Models"
TEMPERATURE = 0.6826

@st.cache_resource(show_spinner="Loading the trained AI models (first visit may take a while)…")
def load_models():
    import torch
    from torchvision import models, transforms
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    def make(arch, filename):
        path = Path(hf_hub_download(repo_id=REPO, filename=filename))
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if all(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k,v in state.items()}
        model = models.efficientnet_b0(weights=None) if arch == "b0" else models.efficientnet_v2_s(weights=None)
        n_classes, in_features = state["classifier.1.weight"].shape
        if n_classes != 5:
            raise ValueError("Expected five flood-severity classes")
        model.classifier[1] = torch.nn.Linear(in_features, n_classes)
        model.load_state_dict(state, strict=True)
        return model.eval()
    b0 = make("b0", "floodsense_e2_advanced.pth")
    v2 = make("v2", "floodsense_v2_finetuned.pth")
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    t0 = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), norm])
    t1 = transforms.Compose([transforms.Resize(384, interpolation=transforms.InterpolationMode.BILINEAR), transforms.CenterCrop(384), transforms.ToTensor(), norm])
    return b0, v2, t0, t1

def predict(image):
    import torch
    b0, v2, t0, t1 = load_models()
    with torch.inference_mode():
        p0 = torch.softmax(b0(t0(image).unsqueeze(0)), dim=1)[0].numpy()
        p1 = torch.softmax(v2(t1(image).unsqueeze(0)), dim=1)[0].numpy()
    raw = np.clip(.7*p0 + .3*p1, 1e-12, 1)
    logits = np.log(raw) / TEMPERATURE
    logits -= logits.max()
    calibrated = np.exp(logits)
    calibrated /= calibrated.sum()
    return calibrated

st.markdown('''<div class="hero"><div class="eyebrow">COMPUTER VISION · RESEARCH PROTOTYPE</div>
<h1>🌊 FloodSense AI</h1><p>Visual assessment of vehicle flood severity, powered by ensemble deep learning.</p></div>''', unsafe_allow_html=True)

intro1,intro2,intro3 = st.columns(3)
with intro1: st.markdown('<div class="smallcard"><div class="kicker">Assessment</div><h3>5 ordered levels</h3><div class="muted">From Level0 to Level4</div></div>', unsafe_allow_html=True)
with intro2: st.markdown('<div class="smallcard"><div class="kicker">AI engine</div><h3>Dual-model ensemble</h3><div class="muted">EfficientNet-B0 + EfficientNetV2-S</div></div>', unsafe_allow_html=True)
with intro3: st.markdown('<div class="smallcard"><div class="kicker">Project status</div><h3>Research beta</h3><div class="muted">Not a certified flood-safety tool</div></div>', unsafe_allow_html=True)

st.write('')
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze image", "📷 Toy-car experiment", "📘 Understanding the levels", "🧪 Research & limitations"])
with tab1:
    left,right = st.columns([1.12,1], gap="large")
    with left:
        st.markdown('<div class="section-title">Upload a vehicle photograph</div>', unsafe_allow_html=True)
        upload = st.file_uploader("Choose a JPG, PNG, or WEBP image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
        if upload:
            try:
                img = ImageOps.exif_transpose(Image.open(upload)).convert("RGB")
                st.image(img, caption="Original aspect ratio preserved", use_container_width=True)
            except Exception:
                st.error("The selected file could not be opened as an image.")
                img = None
        else:
            img = None
            st.info("Upload a flood-affected vehicle image to begin. No example photo is needed.")
        analyze = st.button("Analyze flood severity", type="primary", use_container_width=True, disabled=img is None)
    with right:
        st.markdown('<div class="section-title">AI assessment</div>', unsafe_allow_html=True)
        if analyze and img is not None:
            with st.spinner("Analyzing image…"):
                try:
                    st.session_state["prediction"] = predict(img)
                    st.session_state["image_name"] = upload.name
                except Exception as e:
                    st.error("Unable to load or run the models. Check deployment logs and available memory.")
                    st.caption(f"Technical detail: {type(e).__name__}: {e}")
        if "prediction" in st.session_state:
            probs = st.session_state["prediction"]
            best = int(np.argmax(probs))
            st.markdown(f'<div class="smallcard"><div class="kicker">Predicted flood-severity category</div><div class="bigresult">Level{best}</div><div class="muted">Calibrated model score: <b>{probs[best]:.1%}</b></div></div>', unsafe_allow_html=True)
            st.write('')
            st.markdown("**Probability distribution**")
            for i, val in enumerate(probs):
                c1,c2 = st.columns([1,5])
                with c1: st.caption(f"Level{i}")
                with c2: st.progress(float(val), text=f"{val:.1%}")
            st.caption("Model scores describe relative confidence, not a guaranteed probability of correctness.")
        else:
            st.markdown('<div class="smallcard"><div class="kicker">Awaiting image</div><h3>Your analysis will appear here</h3><div class="muted">Upload a photo and select Analyze flood severity.</div></div>', unsafe_allow_html=True)
    st.write('')
    st.markdown('<div class="notice"><b>Safety notice:</b> This model estimates a dataset-defined visual category, not water depth in centimeters. Never use its prediction to decide whether driving through floodwater is safe.</div>', unsafe_allow_html=True)
with tab2:
    st.subheader("Toy-car flood experiment · Camera Lab")
    st.write("Test whether a model trained on photographs of real flooded vehicles transfers to a small tabletop scene. A wrong prediction is a useful experimental observation, not a failed experiment.")
    st.warning("Use a toy car and a shallow, stable transparent container. Keep water away from phones, chargers and electrical equipment. Have an adult supervise. Never test with a real vehicle in floodwater.")
    st.caption("Take a still photograph using the camera control below. This is not continuous live-video inference.")
    shot = st.camera_input("Capture the toy car and waterline", key="toy_camera")
    toy_upload = st.file_uploader("Or upload a photo taken with another camera", type=["jpg", "jpeg", "png", "webp"], key="toy_upload")
    a, b, c = st.columns(3)
    with a:
        trial_id = st.text_input("Trial ID", value="T01", key="toy_trial_id")
    with b:
        measured_cm = st.number_input("Measured water depth (cm)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="toy_depth")
    with c:
        observer_label = st.selectbox("Observed visual level (optional)", ["Not assigned", "Level0", "Level1", "Level2", "Level3", "Level4"], key="toy_observed")
    notes = st.text_input("Notes (camera angle, lighting, car position)", key="toy_notes")
    source = shot if shot is not None else toy_upload
    if source is not None:
        try:
            toy_img = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
            st.image(toy_img, caption="Experiment photograph", width=480)
            if st.button("Analyze experiment photo", type="primary", key="toy_analyze"):
                with st.spinner("Analyzing experiment photo…"):
                    toy_probs = predict(toy_img)
                best = int(np.argmax(toy_probs))
                st.session_state["toy_last"] = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "trial_id": trial_id,
                    "water_depth_cm": measured_cm,
                    "observed_visual_level": observer_label,
                    "predicted_level": f"Level{best}",
                    "model_score": round(float(toy_probs[best]), 5),
                    **{f"prob_level{i}": round(float(v), 5) for i, v in enumerate(toy_probs)},
                    "notes": notes,
                }
                st.session_state.setdefault("toy_trials", []).append(st.session_state["toy_last"].copy())
        except Exception as e:
            st.error(f"Unable to analyze the image: {type(e).__name__}: {e}")
    if "toy_last" in st.session_state:
        result = st.session_state["toy_last"]
        st.success(f"Predicted {result['predicted_level']} · model score {result['model_score']:.1%}")
        st.caption("Toy-car images are out-of-distribution. This score does not validate the model's prediction or measure depth in centimeters.")
    if st.session_state.get("toy_trials"):
        st.markdown("**Recorded trials in this browser session**")
        st.dataframe(st.session_state["toy_trials"], use_container_width=True)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(st.session_state["toy_trials"][0].keys()))
        writer.writeheader()
        writer.writerows(st.session_state["toy_trials"])
        st.download_button("Download experiment results (CSV)", output.getvalue().encode("utf-8-sig"), "floodsense_toy_trials.csv", "text/csv", use_container_width=True)
        st.caption("Results are stored only in this browser session until downloaded. Photos are not included in the CSV; save them separately with matching trial IDs.")
    st.markdown("**Suggested protocol:** Keep the same car, camera position and lighting. Record a dry scene, then gradually increase water in measured steps. Take at least three photographs per step. Note when the model prediction disagrees with your visual observation. Do not equate the five labels with centimeter ranges.")
with tab3:
    st.subheader("Five ordered flood-severity categories")
    st.write("These are dataset-defined visual labels, not measured water depths. Exact centimeter ranges cannot be inferred from an image alone.")
    cols = st.columns(5)
    for i,c in enumerate(cols):
        with c: st.markdown(f'<div class="level"><strong>Level{i}</strong><p class="muted">{DESCRIPTIONS[i]}</p></div>', unsafe_allow_html=True)
    st.info("The categories describe increasing visual severity. Interpretation may vary with camera angle, vehicle type, lighting, and occlusion.")
with tab4:
    st.subheader("About the research")
    st.write("FloodSense AI investigates whether computer vision can estimate visible vehicle flood-severity categories from photographs. The prototype combines predictions from two trained neural networks and applies post-hoc score calibration.")
    a,b = st.columns(2)
    with a:
        st.markdown("**Current approach**")
        st.write("EfficientNet-B0 (70%) + EfficientNetV2-S (30%). Temperature scaling (T = 0.6826) was fitted using validation data.")
    with b:
        st.markdown("**Evaluation note**")
        st.write("The ensemble achieved approximately 73.5% accuracy on the development validation split. This is not an independent final-test result and does not establish real-world reliability.")
    st.markdown("**Known limitations**")
    st.write("Unusual camera angles, reflections, different vehicles, or images unlike the training data can produce confident but incorrect predictions. The model is not intended for emergency response or road-safety decisions.")
    st.caption("Explainability / Grad-CAM integration is planned; it is not yet included in this deployed version.")

st.divider()
st.caption("FloodSense AI · Educational research prototype · No guarantee of safety or physical water-depth measurement")
