# 🕵️ Deepfake Detector — Hackathon Project

Detect deepfakes with a simple **Streamlit UI** + **FastAPI backend**.  
Lightweight demo using OpenCV + heuristics (blur, artifacts, color), designed to be extended with ML models.

---

## 🚀 Features
- 🖼️ Upload images/videos via **Streamlit frontend**  
- ⚡ **FastAPI backend** with `/predict` endpoint  
- 🔍 Simple rule-based detection (blur, colorfulness, artifacts)  
- 🔧 Easy to extend with pretrained ML deepfake models (XceptionNet, EfficientNet, etc.)  
- ☁️ Deployable on **Render + Streamlit Cloud**  

---

## 🛠️ Setup (Local)

```bash
# Clone repo
git clone https://github.com/<your-username>/deepfake-detector.git
cd deepfake-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
