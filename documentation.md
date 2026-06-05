---
title: DeforestAI - Detection System
author: Shayan Muhammad, Umar Farooq, Owais Ghani Khan
date: 2026
---

<div class="cover-page">
    <div class="cover-header">
        <div class="cover-logo-container">🌲</div>
        <h1 class="cover-title">DeforestAI</h1>
        <p class="cover-subtitle">AI-Powered Satellite Image Analysis for Forest Conservation</p>
        <div class="cover-divider"></div>
    </div>
    
    <div class="cover-info-grid">
        <div class="metadata-card">
            <h3>Academic Metadata</h3>
            <p><strong>Course/Program:</strong> 4th Semester Project (CS A Section)</p>
            <p><strong>Institution:</strong> University of Engineering & Technology (UET), Mardan</p>
            <p><strong>Supervisor:</strong> Sir Shahzad</p>
        </div>
        <div class="metadata-card">
            <h3>Project Authors</h3>
            <p><strong>Shayan Muhammad</strong> (Reg: 24MDBCS557)</p>
            <p><strong>Umar Farooq</strong> (Reg: 24MDBCS561)</p>
            <p><strong>Owais Ghani Khan</strong> (Reg: 24MDBCS587)</p>
        </div>
    </div>
    
    <div class="cover-footer">
        <a href="https://github.com/ShayanMuhammad-CS/DEFORESTAI-DETECTION-SYSTEM" class="cover-github">
            <span class="cover-github-icon">💻</span> GitHub Repository
        </a>
        <p class="cover-date">Session 2024 - 2028 | June 2026</p>
    </div>
</div>

<div class="page-break"></div>

<div class="toc-container">
    <h2>Table of Contents</h2>
    <ul class="toc-list">
        <li class="toc-item">
            <span class="toc-title">1. Abstract & Problem Statement</span>
            <span class="toc-dots"></span>
            <span class="toc-page">3</span>
        </li>
        <li class="toc-item">
            <span class="toc-title">2. Proposed Solution & System Architecture</span>
            <span class="toc-dots"></span>
            <span class="toc-page">3</span>
        </li>
        <li class="toc-item">
            <span class="toc-title">3. Dataset & Preprocessing Pipeline</span>
            <span class="toc-dots"></span>
            <span class="toc-page">4</span>
        </li>
        <li class="toc-item">
            <span class="toc-title">4. Deep Learning Model (MobileNetV2)</span>
            <span class="toc-dots"></span>
            <span class="toc-page">4</span>
        </li>
        <li class="toc-item">
            <span class="toc-title">5. Model Testing & Results</span>
            <span class="toc-dots"></span>
            <span class="toc-page">5</span>
        </li>
        <li class="toc-item">
            <span class="toc-title">6. Explainable AI: Grad-CAM Integration</span>
            <span class="toc-dots"></span>
            <span class="toc-page">6</span>
        </li>
        <li class="toc-item">
            <span class="toc-title">7. Application User Interface (Demo Pics)</span>
            <span class="toc-dots"></span>
            <span class="toc-page">7</span>
        </li>
        <li class="toc-item">
            <span class="toc-title">8. Conclusion & Future Scope</span>
            <span class="toc-dots"></span>
            <span class="toc-page">8</span>
        </li>
    </ul>
</div>

<div class="page-break"></div>

## 1. Abstract & Problem Statement

Deforestation is a critical driver of global climate change, resulting in severe biodiversity loss, disruption of regional water cycles, and massive increases in greenhouse gas emissions. Traditional methods of monitoring forest health—which rely on manual land surveys or basic visual satellite observation—are incredibly slow, labor-intensive, and prone to human error.

**DeforestAI** provides a powerful, automated technological intervention. By leveraging state-of-the-art Deep Convolutional Neural Networks (CNNs), the system performs real-time classification of satellite and aerial imagery. It instantly distinguishes between healthy, intact forest cover and regions degraded by logging or clearing, enabling conservationists, rangers, and policymakers to act before destruction spreads.

---

## 2. Proposed Solution & System Architecture

The project delivers an end-to-end Machine Learning pipeline integrated into a highly accessible web-based interface.

### Architecture Flow
1. **Data Ingestion:** High-resolution satellite images are uploaded by the user via the front-end dashboard.
2. **Preprocessing Engine:** Images undergo automated normalization, resizing (`128x128x3`), and tensor conversion.
3. **Deep Learning Inference:** The fine-tuned MobileNetV2 architecture computes probabilistic classifications.
4. **Explainability Module:** A custom Grad-CAM algorithm visualizes the AI's focal points, drawing a transparent heatmap over the exact regions that influenced the prediction.
5. **Interactive Dashboard:** Built with Streamlit and Plotly, the UI displays real-time confidence gauges, metrics, and actionable risk assessments.

<div class="page-break"></div>

## 3. Dataset & Preprocessing Pipeline

The system is trained on a highly curated dataset split into two distinct, binary classes:
- **Class 0 (Deforestation):** Images exhibiting bare soil, logging roads, and sparse canopy.
- **Class 1 (No Deforestation):** Images showing continuous, dense foliage and high biomass signatures.

### Augmentation Strategy
To prevent model overfitting and simulate diverse environmental conditions, the data pipeline utilizes a real-time `tf.keras.Sequential` augmentation layer. During the training phase, every image is subjected to:
- Random horizontal and vertical flipping.
- Random zooming (up to 30%).
- Random rotations (up to 30%).
- Contrast adjustments (up to 30%).

---

## 4. Deep Learning Model (MobileNetV2)

The core brain of DeforestAI is built upon **MobileNetV2**. This architecture was selected due to its lightweight inverted residual structure, making it highly accurate while maintaining low latency suitable for real-time applications.

### Two-Phase Transfer Learning Strategy
Rather than training a model from scratch, we utilized Transfer Learning, starting with weights pre-trained on ImageNet.

1. **Phase 1: Feature Extraction (Head Training)**
   - The MobileNetV2 base was entirely frozen.
   - A custom classification head (`GlobalAveragePooling2D` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Dense(2, Softmax)`) was attached.
   - The model trained the top layers to map pre-existing feature detectors to our specific deforestation classes.

2. **Phase 2: Fine-Tuning**
   - The top 50 layers of the MobileNetV2 base were unfrozen.
   - The learning rate was exponentially decayed to `1e-4`.
   - The network fine-tuned its deeply embedded feature extractors specifically for the textures of forests and barren land.

<div class="page-break"></div>

## 5. Model Testing & Results

Rigorous evaluation using a split validation dataset proved the model's exceptional capability to correctly identify deforestation threats while minimizing false positives.

### Training Accuracy & Loss
The model demonstrated rapid convergence and high stability across epochs, with minimal divergence between training and validation metrics.

<div class="img-container">
    <img src="plot_2.png" alt="Model Accuracy Plot">
    <div class="img-caption">Figure 1: Training vs Validation Accuracy over epochs.</div>
</div>

<div class="img-container">
    <img src="plot_3.png" alt="Model Loss Plot">
    <div class="img-caption">Figure 2: Training vs Validation Loss showing stable convergence.</div>
</div>

<div class="page-break"></div>

### Confusion Matrix
The confusion matrix highlights the precision and recall of the model across the binary classes, showcasing its reliability in field conditions.

<div class="img-container" style="max-width: 550px; margin: auto;">
    <img src="plot_4.png" alt="Confusion Matrix">
    <div class="img-caption">Figure 3: Confusion Matrix demonstrating classification accuracy across test instances.</div>
</div>

---

## 6. Explainable AI: Grad-CAM Integration

One of the standout features of DeforestAI is its commitment to transparent AI. Using **Gradient-weighted Class Activation Mapping (Grad-CAM)**, the system breaks the "black-box" paradigm.

During inference, `tf.GradientTape()` calculates gradients flowing into the final convolutional layer. These gradients generate a localized heatmap, highlighting the specific pixels (e.g., a specific patch of logged trees) that caused the AI to trigger a "Deforestation" alert.

<div class="callout-box">
    <div class="callout-title">💡 Explainable AI (XAI) in Environmental Audits</div>
    <p class="callout-content">By visualizing why a model made its decision, forestry auditors can instantly verify if the model is tracking genuine canopy clearing or just pathing errors like cloud cover or shadow. This builds immense trust with end-users and researchers.</p>
</div>

<div class="page-break"></div>

## 7. Application User Interface (Demo Pics)

The interactive dashboard enables seamless, real-time predictions with XAI visualization. Below are the functional screenshots of the system in action:

<div class="screenshot-grid">
    <div class="screenshot-card">
        <img src="screenshot_1.png" alt="Demo Pic 1">
        <div class="screenshot-caption">Figure 4.1: Dashboard landing layout and model loading.</div>
    </div>
    <div class="screenshot-card">
        <img src="screenshot_2.png" alt="Demo Pic 2">
        <div class="screenshot-caption">Figure 4.2: Input controls and custom image upload dialog.</div>
    </div>
    <div class="screenshot-card">
        <img src="screenshot_3.png" alt="Demo Pic 3">
        <div class="screenshot-caption">Figure 4.3: Model analyzing healthy forest cover with low risk classification.</div>
    </div>
    <div class="screenshot-card">
        <img src="screenshot_4.png" alt="Demo Pic 4">
        <div class="screenshot-caption">Figure 4.4: High confidence deforestation prediction with alert triggers.</div>
    </div>
    <div class="screenshot-card">
        <img src="screenshot_5.png" alt="Demo Pic 5">
        <div class="screenshot-caption">Figure 4.5: Grad-CAM activation heatmap pointing out logged forest paths.</div>
    </div>
    <div class="screenshot-card">
        <img src="screenshot_6.png" alt="Demo Pic 6">
        <div class="screenshot-caption">Figure 4.6: Metrics visualization and inference confidence details.</div>
    </div>
    <div class="screenshot-card span-full">
        <img src="screenshot_7.png" alt="Demo Pic 7">
        <div class="screenshot-caption">Figure 4.7: Summary reports, statistics panel, and risk assessment indicators.</div>
    </div>
</div>

<div class="page-break"></div>

## 8. Conclusion & Future Scope

### Conclusion
DeforestAI successfully demonstrates that lightweight, modern deep learning architectures can be leveraged for high-impact environmental conservation. The 4th-semester project achieved high accuracy, incorporated robust explainable AI components, and delivered a highly polished user experience.

### Future Scope
- **Multi-Class Segmentation:** Upgrading the model from image classification to pixel-wise semantic segmentation to calculate exact areas of deforested land.
- **Live Satellite API Integration:** Connecting the backend to services like Sentinel-2 or Google Earth Engine for automated, continuous global monitoring.
- **Edge Deployment:** Utilizing TensorFlow Lite to run the application natively on drones for offline, on-site field surveys.

---
<div class="document-footer">
    DeforestAI &copy; 2026 | Developed by Shayan, Umar, and Owais | UET Mardan
</div>
