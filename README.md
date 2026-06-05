<div align="center">
  <h1>🌲 DeforestAI Detection System</h1>
  <p><b>AI-Powered Satellite Image Analysis for Forest Conservation</b></p>
  <p>
    <a href="#features">Features</a> •
    <a href="#system-architecture">Architecture</a> •
    <a href="#installation">Installation</a> •
    <a href="#authors">Authors</a>
  </p>
</div>

---

## 📌 Problem Statement

Deforestation is a critical driver of climate change, resulting in severe biodiversity loss, disruption of water cycles, and increased greenhouse gas emissions. Traditional methods of monitoring forest health—relying on manual land surveys or basic satellite observation—are slow, labor-intensive, and prone to human error.

**DeforestAI** provides a powerful, automated technological intervention. By leveraging state-of-the-art Deep Convolutional Neural Networks (CNNs), the system performs real-time classification of satellite and aerial imagery to instantly distinguish between healthy forest cover and regions degraded by logging or clearing.

---

## ⚙️ System Architecture

The project delivers an end-to-end Machine Learning pipeline integrated into a highly accessible web interface.

1. **Data Ingestion:** High-resolution satellite images are uploaded by the user via the Streamlit dashboard.
2. **Preprocessing Engine:** Images undergo automated normalization, resizing (`128x128x3`), and tensor conversion.
3. **Deep Learning Inference:** A fine-tuned MobileNetV2 architecture computes probabilistic classifications.
4. **Explainability Module:** A custom Grad-CAM algorithm visualizes the AI's focal points, drawing a transparent heatmap over the exact regions that influenced the prediction.
5. **Interactive Dashboard:** Built with Streamlit and Plotly, displaying real-time confidence gauges, metrics, and actionable risk assessments.

---

## 🧠 Deep Learning Model (MobileNetV2)

The core brain of DeforestAI is built upon **MobileNetV2**, utilized for its lightweight inverted residual structure, making it highly accurate while maintaining low latency.

We implemented a **Two-Phase Transfer Learning Strategy**:
1. **Phase 1: Feature Extraction**
   - The MobileNetV2 base was entirely frozen.
   - A custom classification head was attached and trained to map pre-existing feature detectors to our specific deforestation classes.
2. **Phase 2: Fine-Tuning**
   - The top 50 layers of the MobileNetV2 base were unfrozen.
   - The network fine-tuned its deeply embedded feature extractors specifically for the textures of forests and barren land.

---

## 🔍 Explainable AI: Grad-CAM

To build trust with end-users and break the "black-box" paradigm, DeforestAI features a custom implementation of **Gradient-weighted Class Activation Mapping (Grad-CAM)**. 

During inference, gradients are calculated flowing into the final convolutional layer to generate a localized heatmap. This highlights the specific pixels (e.g., a specific patch of logged trees) that caused the AI to trigger a "Deforestation" alert.

---

## 🚀 Installation & Setup

### Prerequisites
Ensure your environment has Python installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### Launching the Application
To start the interactive detection interface:
```bash
streamlit run app.py
```
Access the application locally via your browser at `http://localhost:8501`.

### Retraining the Model
To re-run the 2-phase neural network training with your own dataset:
```bash
python retrain.py --data "/path/to/your/dataset"
```

---

## 👨‍💻 Authors & Academic Details

**4th Semester Project - CS A Section**  
**UET Mardan**  
**Supervised by:** Sir Shahzad

**Developed by:**
- Shayan Muhammad (24MDBCS557)
- Umar Farooq (24MDBCS561)
- Owais Ghani Khan (24MDBCS587)
