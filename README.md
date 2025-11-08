# 🌐 Multimodal Sentiment & Emotional Impact Analyzer

A **multimodal AI Chrome Extension** that analyzes the emotional tone and sentiment of any webpage using **text + image fusion** via pretrained models (**BERT + ResNet50**).  
The backend runs locally on **Flask** and serves results instantly to your Chrome Extension.

---

## 🚀 Features

✅ Extracts **text** and **primary image** from any webpage  
✅ Performs **sentiment analysis** (Positive / Negative / Neutral)  
✅ Detects **emotional intensity** (Low / Medium / High)  
✅ Extracts **top keywords** using **TF-IDF**  
✅ Fuses **text (BERT)** and **image (ResNet50)** features for multimodal representation  
✅ Simple **Chrome popup interface** to analyze any open tab instantly  

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Chrome Extension (HTML, JS, CSS) |
| **Backend API** | Flask (Python) |
| **Text Model** | BERT (`bert-base-uncased`) |
| **Sentiment Model** | DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) |
| **Image Model** | ResNet50 (TensorFlow / Keras) |
| **Keyword Extraction** | TF-IDF (scikit-learn) |

---

## ⚙️ Installation

### 1️⃣ Clone or Download
```bash
git clone https://github.com/yourusername/multimodal-analyzer.git
cd multimodal-analyzer
```
### 2️⃣** Setup Environment**
```bash
python -m venv venv
.\venv\Scripts\activate     # On Windows
# or
source venv/bin/activate    # On Linux/Mac
```
###3️⃣ **Install Dependencies**
```bash
pip install flask torch torchvision torchaudio tensorflow transformers pillow scikit-learn cloudscr
```
###4️⃣ ***Run the Flask API***
```bash
python api_server.py
```

Chrome Extension Setup

Open Chrome and navigate to:

chrome://extensions


Enable Developer Mode

Click Load Unpacked

Select your folder:

multimodal_extension/


Open the popup → Click “Analyze Current Page”


🧠 ***Models Used***
Model	Purpose
BERT	Text Embedding
DistilBERT	Sentiment Detection
ResNet50	Image Embedding
TF-IDF	Keyword Extraction


🛠 ***Libraries Used***
flask
torch
torchvision
torchaudio
tensorflow
transformers
pillow
scikit-learn
cloudscraper
beautifulsoup4
requests
numpy

👨‍💻 ***Authors***

👤 Balanagu Krishna Kiriti
👤 Kethamreddy Vishnu Vardhan Reddy
👤 Koteru Tarun
👤 Pratyush Kumar V
👤 Shakti Narayan

🌟 ***Future Enhancements***

Real-time browser sentiment overlay

Multi-language support

Async & cached inference

Emotion timeline visualization
