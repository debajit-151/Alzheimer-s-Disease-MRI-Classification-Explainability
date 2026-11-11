"""
app.py - Interactive Streamlit Web Application for Alzheimer's MRI Classification
with Explainable AI (Grad-CAM) Visualizer and Performance Dashboard.
"""

import os

import os
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import get_transforms
from src.models.custom_cnn import AlzheimerCustomCNN
from src.models.transfer_model import AlzheimerResNet
from src.explainability import GradCAM, overlay_cam_on_image

# Page configuration
st.set_page_config(
    page_title="Alzheimer's MRI Diagnostic AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

CLASS_DESCRIPTIONS = {
    "Non_Demented": {
        "badge": "🟢 Normal / Cognitively Intact",
        "desc": "Brain parenchyma exhibits normal volume, preserved cortical thickness, and normal ventricular space without observable hippocampal atrophy."
    },
    "Very_Mild_Demented": {
        "badge": "🟡 Very Mild Impairment",
        "desc": "Early sub-clinical changes. Sub-threshold atrophy in the entorhinal cortex and medial temporal lobe with marginal ventricular widening."
    },
    "Mild_Demented": {
        "badge": "🟠 Mild Impairment",
        "desc": "Noticable bilateral medial temporal lobe atrophy, evident enlargement of lateral ventricles, and moderate sulcal widening."
    },
    "Moderate_Demented": {
        "badge": "🔴 Moderate / Severe Dementia",
        "desc": "Pronounced global cerebral atrophy, substantial ventricular dilation, severe bilateral hippocampal volume loss, and deep sulcal enlargement."
    }
}

@st.cache_resource
def load_cached_model(model_type, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint["class_names"]

    if model_type == "custom_cnn":
        model = AlzheimerCustomCNN(num_classes=len(class_names))
    elif model_type == "resnet18":
        model = AlzheimerResNet(num_classes=len(class_names), pretrained=False)
    else:
        raise ValueError("Invalid model type")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, device

def main():
    st.markdown("<div class='main-header'>🧠 Alzheimer's Disease MRI Classification & Explainability</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Deep Learning Diagnostic Assistant featuring Pretrained Residual Networks & Grad-CAM Visual Heatmaps</div>", unsafe_allow_html=True)

    # Sidebar: Model Selection & Settings
    st.sidebar.header("⚙️ Model Configuration")
    
    models_dir = Path("outputs/models")
    custom_ckpt = models_dir / "custom_cnn_best.pth"
    resnet_ckpt = models_dir / "resnet18_best.pth"

    available_models = {}
    if resnet_ckpt.exists():
        available_models["Transfer Learning (ResNet-18)"] = ("resnet18", resnet_ckpt)
    if custom_ckpt.exists():
        available_models["Custom 4-Block CNN Baseline"] = ("custom_cnn", custom_ckpt)

    if not available_models:
        st.error("⚠️ No trained model checkpoints found in `outputs/models/`. Please execute `python train.py --model all` first!")
        st.stop()

    selected_model_label = st.sidebar.selectbox("Choose Model Architecture:", list(available_models.keys()))
    m_type, ckpt_path = available_models[selected_model_label]

    model, class_names, device = load_cached_model(m_type, str(ckpt_path))

    cam_alpha = st.sidebar.slider("Grad-CAM Overlay Transparency (α):", min_value=0.1, max_value=0.9, value=0.55, step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Clinical Stage Reference")
    for k, v in CLASS_DESCRIPTIONS.items():
        st.sidebar.markdown(f"**{k.replace('_', ' ')}**: {v['badge']}")

    # Tabs
    tab_diagnostics, tab_benchmarks, tab_about = st.tabs(["🩺 MRI Diagnostic & Grad-CAM", "📊 Model Evaluation Benchmarks", "📖 Clinical & Technical Details"])

    with tab_diagnostics:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("1. Input Brain MRI Scan")
            input_mode = st.radio("Choose Input Method:", ["Select Sample MRI", "Upload MRI Image"])

            image_to_process = None
            sample_label = None

            if input_mode == "Select Sample MRI":
                samples_dir = Path("data/samples")
                if samples_dir.exists():
                    sample_files = list(samples_dir.glob("*.jpg"))
                    if sample_files:
                        sample_names = [f.stem for f in sample_files]
                        chosen_idx = st.selectbox("Pick a pre-loaded sample scan:", range(len(sample_names)), format_func=lambda i: sample_names[i])
                        chosen_path = sample_files[chosen_idx]
                        image_to_process = Image.open(chosen_path).convert("RGB")
                        sample_label = sample_names[chosen_idx]
                else:
                    st.warning("Sample scans folder `data/samples` not found.")

            else:
                uploaded_file = st.file_uploader("Upload an axial brain MRI scan (JPG/PNG)", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    image_to_process = Image.open(uploaded_file).convert("RGB")

            if image_to_process is not None:
                st.image(image_to_process, caption=f"Selected MRI Scan ({sample_label if sample_label else 'Uploaded'})", use_container_width=True)

        with col2:
            st.subheader("2. Diagnostic Prediction & Grad-CAM")
            if image_to_process is not None:
                with st.spinner("Analyzing brain neuroimaging slice..."):
                    # Preprocess
                    _, eval_transform = get_transforms(img_size=(128, 128))
                    input_tensor = eval_transform(image_to_process).unsqueeze(0).to(device)

                    # Grad-CAM Engine
                    cam_engine = GradCAM(model, model.get_cam_target_layer())
                    with torch.enable_grad():
                        heatmap, target_idx = cam_engine.generate_heatmap(input_tensor)

                    # Inference
                    with torch.no_grad():
                        logits = model(input_tensor)
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                    pred_class = class_names[target_idx]
                    confidence = probs[target_idx] * 100

                    # Display top prediction
                    info = CLASS_DESCRIPTIONS.get(pred_class, {"badge": pred_class, "desc": ""})
                    st.success(f"### Predicted Diagnosis: **{pred_class.replace('_', ' ')}**")
                    st.markdown(f"**Clinical Status:** {info['badge']}")
                    st.markdown(f"**Model Confidence:** `{confidence:.2f}%`")
                    st.info(f"**Neuro-Radiological Insight:** {info['desc']}")

                    # Class Probabilities Bar Chart
                    st.markdown("#### Probability Distribution:")
                    for c_idx, c_name in enumerate(class_names):
                        prob_val = float(probs[c_idx])
                        st.write(f"**{c_name.replace('_', ' ')}**: {prob_val * 100:.1f}%")
                        st.progress(prob_val)

                    # Explainability Visualizer
                    st.markdown("---")
                    st.subheader("3. Explainable AI: Grad-CAM Activation Heatmap")
                    st.caption("Warm regions (Red/Yellow) highlight anatomical regions most influential in determining the clinical stage.")

                    orig_np = np.array(image_to_process.resize((128, 128)))
                    overlay, _ = overlay_cam_on_image(orig_np, heatmap, alpha=cam_alpha)

                    cam_col1, cam_col2 = st.columns(2)
                    with cam_col1:
                        st.image(heatmap, caption="Raw Grad-CAM Feature Map", use_container_width=True, clamp=True)
                    with cam_col2:
                        st.image(overlay, caption="Anatomical Overlay (MRI + Heatmap)", use_container_width=True)

            else:
                st.info("👈 Please select or upload a Brain MRI scan on the left to begin diagnosis.")

    with tab_benchmarks:
        st.subheader("📊 Model Evaluation & Benchmarks")
        figures_dir = Path("outputs/figures")

        report_file = Path("outputs/evaluation_report.json")
        if report_file.exists():
            import json
            with open(report_file, "r") as f:
                report_data = json.load(f)
            
            st.markdown("#### Overall Test Set Performance (1,280 Unseen Images)")
            cols = st.columns(len(report_data))
            for i, (m_name, m_stats) in enumerate(report_data.items()):
                with cols[i]:
                    st.markdown(f"### {m_name}")
                    st.metric("Test Accuracy", f"{m_stats['accuracy']*100:.2f}%")
                    st.metric("Balanced Accuracy", f"{m_stats['balanced_accuracy']*100:.2f}%")
                    st.metric("Macro F1-Score", f"{m_stats['macro_f1']*100:.2f}%")
                    st.metric("Weighted F1-Score", f"{m_stats['weighted_f1']*100:.2f}%")

        comp_chart = figures_dir / "model_comparison_benchmark.png"
        if comp_chart.exists():
            st.image(str(comp_chart), caption="Architectural Benchmark Comparison", use_container_width=True)

        st.markdown("---")
        st.subheader("Confusion Matrices & ROC Curves")
        c1, c2 = st.columns(2)
        with c1:
            cm_file = figures_dir / f"{m_type}_confusion_matrix.png"
            if cm_file.exists():
                st.image(str(cm_file), caption=f"{selected_model_label} - Confusion Matrix", use_container_width=True)
        with c2:
            roc_file = figures_dir / f"{m_type}_roc_curves.png"
            if roc_file.exists():
                st.image(str(roc_file), caption=f"{selected_model_label} - Multi-Class ROC Curves", use_container_width=True)

        gallery_file = figures_dir / f"{m_type}_gradcam_gallery.png"
        if gallery_file.exists():
            st.markdown("---")
            st.subheader("Grad-CAM Explainability Gallery Across All Stages")
            st.image(str(gallery_file), caption="Explainability Gallery: Test MRI vs Attention Heatmaps", use_container_width=True)

    with tab_about:
        st.subheader("📖 Project Architecture & Medical Methodology")
        st.markdown("""
        ### Problem Statement
        Alzheimer's Disease (AD) is a progressive neurodegenerative disorder leading to severe cognitive decline and memory loss.
        Early diagnosis is crucial for intervention. Magnetic Resonance Imaging (MRI) non-invasively reveals characteristic biomarkers:
        - **Bilateral hippocampal atrophy**
        - **Ventricular dilation / enlargement**
        - **Cortical sulcal widening and cerebral volume reduction**

        ### Dataset Specification
        - **Gold-Standard Alzheimer's 4-Class MRI Dataset**: 6,400 T1-weighted axial brain MRI slices.
        - **Clinical Stages**:
          1. `Non_Demented`: Cognitively intact healthy control.
          2. `Very_Mild_Demented`: Early stage mild cognitive impairment (MCI).
          3. `Mild_Demented`: Evident dementia with visible ventricular expansion.
          4. `Moderate_Demented`: Advanced neurodegeneration with profound atrophy.

        ### Deep Learning Models
        1. **Custom Deep CNN Baseline**:
           - 4 convolutional blocks with increasing filter depths (32 -> 64 -> 128 -> 256).
           - Batch Normalization in each block for gradient stabilization.
           - Progressive Spatial Dropout and Dense Dropout for regularizing minority classes.
           - Resolution-invariant Global Average Pooling.
        2. **Transfer Learning (ResNet-18)**:
           - Pre-trained on ImageNet with deep residual connections ($F(x) + x$).
           - Eliminates vanishing gradients in deep feature extraction.
           - Tailored multi-stage classification head with fine-tuned learning rates.

        ### Tackling Class Imbalance
        - Inverse class weighting applied in `CrossEntropyLoss`:
          $$w_c = \\frac{N_{total}}{N_{classes} \\times N_c}$$
        - Ensures minority classes (e.g., *Moderate Demented*) are not overwhelmed by majority classes.

        ### Explainable AI (Grad-CAM)
        Gradient-weighted Class Activation Mapping computes the gradient of the class score with respect to feature maps of the final convolutional layer:
        $$\\alpha_k^c = \\frac{1}{Z} \\sum_i \\sum_j \\frac{\\partial Y^c}{\\partial A_{ij}^k}$$
        $$L_{Grad-CAM}^c = \\text{ReLU}\\left(\\sum_k \\alpha_k^c A^k\\right)$$
        This guarantees transparency and clinical verification of deep learning predictions.
        """)

if __name__ == "__main__":
    main()
