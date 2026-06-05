---
title: DeforestAI - Detection System
author: Shayan Muhammad, Umar Farooq, Owais Ghani Khan
date: 2026
---

<div class="cover-page">
    <div class="logo">🌲</div>
    <h1>DeforestAI</h1>
    <p class="subtitle">AI-Powered Satellite Image Analysis for Forest Conservation</p>
    <div class="divider"></div>
    <p class="author">Developed by:</p>
    <p class="author">Shayan Muhammad (24MDBCS557)<br>
    Umar Farooq (24MDBCS561)<br>
    Owais Ghani Khan (24MDBCS587)</p>
    <p class="date">Final Year Project &middot; 2026</p>
</div>

\pagebreak

## 1. Executive Summary

Deforestation is one of the most pressing environmental crises of our time, contributing heavily to climate change and biodiversity loss. **DeforestAI** is an advanced machine learning system that leverages state-of-the-art deep learning to automatically detect deforestation from satellite and aerial imagery. 

By enabling conservationists, governments, and NGOs to monitor forest health at scale and with unprecedented speed, DeforestAI acts as a critical technological intervention in the fight to preserve global forests.

---

## 2. System Architecture

The DeforestAI system is composed of two primary components:
1. **The Core AI Model:** A fine-tuned MobileNetV2 Convolutional Neural Network (CNN) trained specifically on top-down satellite imagery to classify land into "Deforested" or "Healthy Forest" regions.
2. **The Production Interface:** A robust, real-time web application built with Streamlit, Plotly, and TensorFlow, enabling non-technical users to upload satellite images and receive instant, interpretable AI diagnostics.

### High-Level Data Flow
1. **Input:** User uploads a high-resolution satellite or aerial image (JPG, PNG).
2. **Preprocessing:** Image is automatically rescaled to `128x128` pixels and converted to raw RGB arrays.
3. **Inference:** The AI model processes the image to compute class probabilities.
4. **Explainability:** The system generates a Gradient-weighted Class Activation Mapping (Grad-CAM) heatmap.
5. **Output:** The UI presents the verdict, confidence scores, and visual overlays highlighting critical regions.

---

## 3. Data Processing & Augmentation

High-quality data is the lifeblood of deep learning. The dataset is organized into two distinct classes:
- `Deforestation` (Class 0): Characterized by bare soil, logging roads, or reduced canopy density.
- `No Deforestation` (Class 1): Intact, dense forest cover with continuous canopy.

### Preprocessing
All images are uniformly resized to `128x128x3` (RGB) to match the input requirements of the neural network. A `Rescaling` layer of `1./255` is embedded directly into the neural network graph to normalize pixel values from `[0, 255]` to `[0, 1]`.

### Data Augmentation
To improve the model's generalization capabilities and prevent overfitting on a limited dataset, we apply real-time data augmentation during the training pipeline:
- **Random Flips:** Horizontal and vertical mirroring.
- **Random Rotations:** Up to 20% rotation.
- **Random Zoom & Contrast:** Adjusting scale and contrast up to 20%.

---

## 4. Deep Learning Model Architecture

DeforestAI utilizes **MobileNetV2** as its foundational backbone—chosen for its exceptional balance between high accuracy and computational efficiency, making it ideal for real-time deployment.

### Two-Phase Fine-Tuning Strategy
Training was conducted using a rigorous, two-phase transfer learning methodology to preserve pre-trained ImageNet features while adapting the model to specific environmental topographies.

**Phase 1: Feature Extraction**
- **Base Model:** MobileNetV2 (frozen).
- **Head:** GlobalAveragePooling2D $\rightarrow$ Dense(128, ReLU) $\rightarrow$ BatchNormalization $\rightarrow$ Dropout(0.4) $\rightarrow$ Dense(2, Softmax).
- **Optimization:** Adam optimizer with learning rate of `1e-3`.
- **Duration:** 10 epochs.

**Phase 2: Fine-Tuning**
- **Process:** The top 50 layers of the MobileNetV2 base model are unfrozen to allow fine-grained adaptation to the specific textures of foliage and bare land.
- **Optimization:** Learning rate drastically reduced to `1e-4` to prevent destructive updates.
- **Duration:** 20 epochs.

*Training utilized an advanced custom callback system (`BestModelSaver`) leveraging pure NumPy memory management to ensure the most robust model weights were perfectly restored without serialization errors.*

---

## 5. Application Interface

The front-end is powered by **Streamlit**, deeply customized with CSS to provide a premium, dark-green environmental aesthetic. 

### Key Features
- **Interactive Dashboard:** Upload interfaces equipped with dynamic statistics and meta-data extraction.
- **Real-Time Analytics:** Powered by `Plotly` gauge charts and horizontal bar graphs, communicating AI confidence clearly.
- **Model Transparency:** A dedicated Model Information page parsing the exact architecture and parameters directly from the loaded `deforestation.h5` model.

---

## 6. AI Explainability: Grad-CAM

Deep learning models are notoriously "black boxes." To build trust with conservationists, DeforestAI features a custom implementation of **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

### How It Works:
1. The AI extracts feature maps from the final convolutional layer (the input to the GlobalAveragePooling layer).
2. `tf.GradientTape()` calculates the gradients of the predicted class score with respect to these feature maps.
3. The gradients are pooled and used to weight the feature maps, highlighting the specific regions of the input image that most heavily influenced the AI's final decision.
4. This heatmap is colorized using a highly optimized, pure-NumPy Jet colormap and superimposed transparently over the original satellite imagery.

---

## 7. Setup & Execution Guide

### Prerequisites
Ensure your environment meets the dependencies specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Key libraries: `tensorflow>=2.13.0`, `streamlit>=1.32.0`, `plotly`, `pillow`)*

### Retraining the Model
To re-run the 2-phase neural network training with your own dataset:
```bash
python retrain.py --data "/path/to/your/dataset"
```
The script will output `deforestation.h5` containing the optimal weights.

### Launching the Application
To start the interactive detection interface:
```bash
streamlit run app.py
```
Access the application locally via your browser at `http://localhost:8501`.

---
<div class="footer">
    <p>DeforestAI Documentation &copy; 2026. Designed for Environmental Impact.</p>
</div>
