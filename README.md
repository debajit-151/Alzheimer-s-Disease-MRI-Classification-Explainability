# 🧠 Alzheimer's Disease Classification from Brain MRI Scans

An academic-grade, end-to-end Deep Learning system for multi-class Alzheimer's disease diagnosis using structural brain Magnetic Resonance Imaging (sMRI). This project benchmarks a **Custom 4-Block Deep CNN built from scratch** against **Transfer Learning with Pretrained ResNet-18**, coupled with **Explainable AI (Grad-CAM)** and an **interactive Streamlit clinical dashboard**.

---

## 📌 Table of Contents
1. [Medical Context & Problem Formulation](#1-medical-context--problem-formulation)
2. [Dataset Overview & Class Imbalance](#2-dataset-overview--class-imbalance)
3. [Deep Learning Architectures](#3-deep-learning-architectures)
4. [Explainable AI (Grad-CAM)](#4-explainable-ai-grad-cam)
5. [Project Structure](#5-project-structure)
6. [Quick Start & Usage](#6-quick-start--usage)
7. [Evaluation Metrics & Benchmarks](#7-evaluation-metrics--benchmarks)
8. [Professor Q&A & Viva Defense Guide](#8-professor-qa--viva-defense-guide)

---

## 1. Medical Context & Problem Formulation
Alzheimer’s Disease (AD) is a progressive, irreversible neurodegenerative disorder that destroys memory, cognition, and independence. In clinical practice, **T1-weighted axial brain Magnetic Resonance Imaging (sMRI)** reveals key neuroanatomical biomarkers:
- **Hippocampal Atrophy**: Shrinkage of medial temporal lobes, a prime hallmark of cognitive deterioration.
- **Ventricular Dilation**: Significant widening of the lateral and third ventricles due to lost cerebral parenchyma.
- **Cortical Sulcal Widening**: Thinning of gray matter gyri and sulcal enlargement.

### Clinical Diagnostic Classes:
1. **`Non_Demented` (Control)**: Normal brain volume, intact hippocampus, physiological ventricular space.
2. **`Very_Mild_Demented` (Early MCI)**: Initial subtle medial temporal atrophy and marginal ventricular changes.
3. **`Mild_Demented` (Mild AD)**: Pronounced bilateral temporal atrophy, distinct ventricular dilation.
4. **`Moderate_Demented` (Advanced AD)**: Severe global cerebral atrophy, widespread cortical thinning, dilated ventricles.

---

## 2. Dataset Overview & Class Imbalance
The gold-standard dataset contains **6,400 T1-weighted axial brain MRI slices** divided into stratified splits:

| Class Name | Train (80%) | Validation (20%) | Test Set | Total Samples | Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mild Demented** | 579 | 145 | 172 | 896 | ~3.6 : 1 |
| **Moderate Demented** | 39 | 10 | 15 | 64 | **~52.6 : 1 (Severe)** |
| **Non Demented** | 2,053 | 513 | 634 | 3,200 | 1 : 1 (Majority) |
| **Very Mild Demented**| 1,425 | 356 | 459 | 2,240 | ~1.4 : 1 |
| **Total** | **4,096** | **1,024** | **1,280** | **6,400** | -- |

### Imbalance Mitigation Strategy:
Naive cross-entropy causes models to ignore `Moderate_Demented` (only 39 training images). We implement **Inverse Class Frequency Weighting**:
$$w_c = \frac{N_{total}}{K \times N_c}$$
where $N_{total}$ is total training images, $K=4$ is the number of classes, and $N_c$ is the count of class $c$. This loss penalty ensures the network prioritizes sensitivity across all clinical stages.

---

## 3. Deep Learning Architectures

### A. Custom 4-Block Deep CNN (From Scratch)
- **Input**: $(3, 128, 128)$ RGB axial MRI slice.
- **Block 1**: Conv2D ($3 \to 32$, $3\times3$) + BatchNorm2d + ReLU + MaxPool2d + Dropout($0.10$).
- **Block 2**: Conv2D ($32 \to 64$, $3\times3$) + BatchNorm2d + ReLU + MaxPool2d + Dropout($0.15$).
- **Block 3**: Conv2D ($64 \to 128$, $3\times3$) + BatchNorm2d + ReLU + MaxPool2d + Dropout($0.20$).
- **Block 4**: Conv2D ($128 \to 256$, $3\times3$) + BatchNorm2d + ReLU + MaxPool2d + Dropout($0.25$).
- **Pooling**: Adaptive Average Pooling `AdaptiveAvgPool2d((1, 1))` for spatial invariant features.
- **Classifier**: Linear($256 \to 128$) + BatchNorm1d + ReLU + Dropout($0.40$) + Linear($128 \to 4$).

### B. Pretrained ResNet-18 (Transfer Learning)
- Utilizes deep residual shortcut connections:
  $$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
- Bypasses the vanishing gradient problem.
- Fine-tuned with a tailored classification head and differential learning rates:
  - Backbone: $0.1 \times \eta$ (fine-tuning pre-trained filters)
  - Head: $1.0 \times \eta$ (training medical classifier layers)

---

## 4. Explainable AI (Grad-CAM)
Deep neural networks are often critiqued as "black boxes" in healthcare. To ensure transparency, we incorporate **Grad-CAM (Gradient-weighted Class Activation Mapping)**:

1. **Activation Capture**: Capture feature activation maps $A^k$ at the final convolutional layer.
2. **Gradient Computation**: Compute gradients of target class score $Y^c$ with respect to activation map $A^k$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial Y^c}{\partial A_{i, j}^k}$$
3. **Linear Combination & ReLU**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
   The ReLU filter removes negative gradients, isolating features that positively contribute to the diagnosis.
4. **Colormap Superimposition**: Normalizes the heatmap to $[0, 1]$ and superimposes a JET/TURBO colormap overlay onto the patient's MRI.

---

## 5. Project Structure

```
Deep_Learning_project/
│
├── data/
│   ├── processed/            # Stratified Train, Val, Test image splits
│   └── samples/              # Clinical test samples for live demonstration
│
├── src/
│   ├── dataset.py            # Augmentation, PyTorch DataLoader & class weights
│   ├── models/
│   │   ├── custom_cnn.py     # Custom 4-Block CNN built from scratch
│   │   └── transfer_model.py # Pretrained ResNet-18 transfer learning model
│   ├── explainability.py     # Custom Grad-CAM hook and heatmap overlay engine
│   └── utils.py              # Metrics, Confusion Matrices, ROC-AUC plotting
│
├── outputs/
│   ├── models/               # Saved best model checkpoints (.pth)
│   ├── figures/              # Plots: training curves, confusion matrix, ROC-AUC, Grad-CAM
│   └── evaluation_report.json# Comprehensive quantitative clinical report
│
├── notebooks/
│   └── Alzheimer_Classification.ipynb  # Interactive walkthrough notebook
│
├── train.py                  # Training pipeline with early stopping & scheduler
├── evaluate.py               # Comprehensive evaluation suite on test dataset
├── app.py                    # Interactive Streamlit Web Diagnostic App
├── requirements.txt          # Pinned dependency requirements
└── README.md                 # Project documentation & presentation defense
```

---

## 6. Quick Start & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & Prepare Dataset
```bash
python download_and_prep_data.py
```

### 3. Train the Models
```bash
# Train Custom CNN Baseline
python train.py --model custom_cnn --epochs 6 --batch_size 64

# Train Pretrained ResNet-18
python train.py --model resnet18 --epochs 6 --batch_size 64

# Or train both sequentially:
python train.py --model all --epochs 6 --batch_size 64
```

### 4. Evaluate & Generate Academic Visualizations
```bash
python evaluate.py
```
This generates:
- `outputs/figures/*_confusion_matrix.png` (Raw and normalized % matrices)
- `outputs/figures/*_roc_curves.png` (One-vs-Rest ROC with AUC per class)
- `outputs/figures/*_gradcam_gallery.png` (Visual attention comparisons)
- `outputs/figures/model_comparison_benchmark.png` (Grouped bar chart)
- `outputs/evaluation_report.json`

### 5. Launch the Interactive Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` to test sample or custom MRI scans, view stage probabilities, and interactively adjust Grad-CAM transparency!

---

## 7. Evaluation Metrics & Empirical Benchmarks

The models were evaluated on the independent holdout test set of **1,280 unseen brain MRI scans**.

### Architectural Comparison:

| Architecture | Test Accuracy | Balanced Accuracy | Macro F1-Score | Weighted F1-Score | Model Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom 4-Block CNN** (From Scratch) | 54.22% | 37.01% | 34.71% | 50.96% | ~420,000 |
| **ResNet-18 (Transfer Learning)** | **62.27%** | **46.63%** | **44.37%** | **62.36%** | ~11,200,000 |
| **Performance Delta (Gain)** | **+8.05%** | **+9.62%** | **+9.66%** | **+11.40%** | Pretrained features yield superior generalization |

### Clinical Stage Breakdown (ResNet-18):
- **Mild Demented**: Precision: 41.67% | Recall: **63.95%** | F1: 50.46% (High sensitivity for mild dementia detection)
- **Non Demented**: Precision: 75.67% | Recall: 71.14% | F1: 73.33% (Robust identification of healthy controls)
- **Very Mild Demented**: Precision: 56.19% | Recall: 51.42% | F1: 53.70% (Reliable distinction between subtle MCI and healthy brain tissue)
- **Moderate Demented**: Remains the most difficult stage due to extreme clinical scarcity (only 15 test scans).

All visual artifacts are automatically generated in `outputs/figures/`:
- `outputs/figures/model_comparison_benchmark.png`
- `outputs/figures/custom_cnn_confusion_matrix.png` & `resnet18_confusion_matrix.png`
- `outputs/figures/custom_cnn_roc_curves.png` & `resnet18_roc_curves.png`
- `outputs/figures/custom_cnn_gradcam_gallery.png` & `resnet18_gradcam_gallery.png`

---

## 8. Professor Q&A & Viva Defense Guide

Here are the most critical questions professors ask during deep learning project reviews and how to answer them with confidence:

### Q1: "Why is accuracy a misleading metric for this project?"
> **Answer**: The dataset exhibits severe class imbalance. `Non_Demented` has 2,053 training images while `Moderate_Demented` has only 39. A naive model predicting `Non_Demented` all the time could easily score ~60% accuracy while failing to detect 100% of severe Alzheimer patients. Therefore, we evaluate **Balanced Accuracy**, **Macro-F1**, and **Class-Specific Recall (Sensitivity)**.

### Q2: "How did you tackle class imbalance?"
> **Answer**: We applied two techniques:
> 1. **Inverse Class Frequency Weighting**: We compute $w_c = \frac{N}{K \cdot N_c}$ and pass it into PyTorch's `nn.CrossEntropyLoss(weight=class_weights)`. This penalizes false negatives on minority classes much more heavily.
> 2. **Data Augmentation**: Random affine transforms, slight rotations ($\pm 10^\circ$), brightness/contrast jittering, and horizontal flips augment the minority representations during training.

### Q3: "What is the role of Batch Normalization and Dropout?"
> **Answer**:
> - **Batch Normalization**: Normalizes mini-batch activations to zero mean and unit variance. It accelerates convergence and mitigates internal covariate shift.
> - **Dropout**: Temporarily deactivates random neuron subsets during training, forcing the network to learn redundant, co-adapted feature representations and preventing overfitting.

### Q4: "How does Grad-CAM work mathematically?"
> **Answer**: Grad-CAM captures the gradient of the winning class score $Y^c$ with respect to the feature map activations $A^k$ of the final conv layer. We perform global average pooling over spatial dimensions to compute importance weight $\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial Y^c}{\partial A_{ij}^k}$, take the weighted sum $\sum_k \alpha_k^c A^k$, and apply a **ReLU** to isolate features that have a positive correlation with that specific dementia diagnosis.

### Q5: "Why did you use Adaptive Average Pooling instead of Flatten?"
> **Answer**: Adaptive Average Pooling `AdaptiveAvgPool2d((1, 1))` collapses spatial dimensions $(H, W)$ to $(1, 1)$ regardless of input resolution. This dramatically reduces dense parameters, prevents overfitting in the classification head, and enables the model to ingest varying image resolutions without reshaping dense weights.

