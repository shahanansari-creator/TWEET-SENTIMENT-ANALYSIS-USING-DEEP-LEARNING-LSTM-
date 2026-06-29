# TWEET-SENTIMENT-ANALYSIS-USING-DEEP-LEARNING-LSTM-

# 🐦 Tweet Sentiment Analysis using Deep Learning (LSTM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green)](https://www.nltk.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> A deep learning pipeline that performs **multi-class sentiment analysis** on political tweets using an LSTM neural network — classifying each tweet as **Positive**, **Negative**, or **Neutral**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Pipeline Workflow](#-pipeline-workflow)
- [Contributing](#-contributing)


---

## 🔍 Overview

This project builds an end-to-end NLP pipeline to analyze the sentiment of tweets related to political candidates. Given the noisy, informal nature of tweet text, the pipeline applies rigorous text preprocessing before feeding data into a multi-layer deep learning model capable of distinguishing between positive, negative, and neutral sentiments.

**Key highlights:**
- Full preprocessing pipeline (stopword removal, URL stripping, punctuation cleaning)
- Word-level tokenization and sequence padding
- LSTM-based deep learning model with dropout regularization
- Training with early stopping and learning rate scheduling
- Confusion matrix and classification report for evaluation
- Ready-to-use inference function for new tweets

---

## 📁 Project Structure

```
tweet-sentiment-analysis/
│
├── tweet_sentiment_analysis.py   # Main Colab notebook / script
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
│
├── data/
│   └── tweets.csv                # Input dataset (candidate, sentiment, text)
│
├── outputs/
│   ├── tweet_sentiment_lstm.keras   # Saved trained model
│   ├── sentiment_distribution.png   # Class distribution chart
│   ├── training_history.png         # Accuracy & loss curves
│   └── confusion_matrix.png         # Evaluation heatmap
│
└── notebooks/
    └── tweet_sentiment_analysis.ipynb  # Jupyter/Colab notebook version
```

---

## 📊 Dataset

The model expects a CSV file with at least the following three columns:

| Column | Description |
|---|---|
| `candidate` | Name of the political candidate the tweet refers to |
| `sentiment` | Ground-truth label — `Positive`, `Negative`, or `Neutral` |
| `text` | Raw tweet content |

**Example rows:**

```
candidate,sentiment,text
Candidate_A,Positive,"I love this candidate! Great policies and strong leadership."
Candidate_B,Negative,"Terrible debate performance, very disappointed."
Candidate_A,Neutral,"Candidate A spoke at the rally today in downtown."
```

> If you are using a custom dataset, ensure it contains these three columns. Any additional columns are automatically ignored during preprocessing.

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| TensorFlow / Keras | 2.x | Model building & training |
| NLTK | 3.x | Stopword removal & tokenization |
| scikit-learn | 1.x | Label encoding, train-test split, metrics |
| pandas | 2.x | Data loading and manipulation |
| NumPy | 1.x | Numerical operations |
| Matplotlib / Seaborn | latest | Visualization |

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────────┐
│         Input (padded sequences)        │
│              max_len = 100              │
└────────────────────┬────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   Embedding Layer   │
          │  vocab=10000, dim=64│
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  SpatialDropout1D   │
          │      rate = 0.3     │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │     LSTM Layer      │
          │  units=128          │
          │  dropout=0.2        │
          │  recurrent_drop=0.2 │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Dense (64, ReLU)  │
          │   + Dropout (0.3)   │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Dense (3, Softmax) │
          │  Output Layer       │
          └─────────────────────┘
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding | (None, 100, 64) | 640,000 |
| SpatialDropout1D | (None, 100, 64) | 0 |
| LSTM | (None, 128) | 98,816 |
| Dense + Dropout | (None, 64) | 8,256 |
| Dense (Output) | (None, 3) | 195 |

**Loss Function:** Categorical Cross-Entropy  
**Optimizer:** Adam (lr = 1e-3)  
**Metric:** Accuracy

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tweet-sentiment-analysis.git
cd tweet-sentiment-analysis
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### `requirements.txt`

```
tensorflow>=2.10.0
pandas>=1.5.0
numpy>=1.23.0
nltk>=3.7
scikit-learn>=1.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## 🚀 Usage

### Option A — Google Colab (Recommended)

1. Open the notebook in Colab using the badge at the top.
2. Upload your `tweets.csv` when prompted, or use the built-in synthetic data.
3. Run all cells sequentially (`Runtime → Run all`).

### Option B — Run Locally

```bash
python tweet_sentiment_analysis.py
```

Make sure your CSV is placed at `data/tweets.csv`, or update the path inside the script.

### Predict on a New Tweet

```python
from tweet_sentiment_analysis import predict_sentiment

result = predict_sentiment("This candidate has the best economic plan I've ever seen!")
print(result)
# Output → Sentiment: Positive  |  Confidence: 94.2%
```

### Load a Saved Model

```python
from tensorflow.keras.models import load_model

model = load_model("outputs/tweet_sentiment_lstm.keras")
```

---

## 📈 Results

### Training History

| Metric | Training | Validation |
|---|---|---|
| Accuracy | ~94% | ~89% |
| Loss | ~0.18 | ~0.31 |

> Results vary depending on dataset size and class balance. The synthetic demo dataset yields high accuracy due to its small size; results on real-world noisy tweet data will differ.

### Sample Predictions

| Tweet | Predicted Sentiment | Confidence |
|---|---|---|
| "Absolutely love this candidate! Best policies ever." | ✅ Positive | 96.1% |
| "Can't stand this politician. They ruin everything." | ❌ Negative | 91.4% |
| "The candidate gave a speech at the convention today." | ➖ Neutral | 88.7% |

---

## 🔄 Pipeline Workflow

```
Raw CSV Data
     │
     ▼
Select Columns (candidate, sentiment, text)
     │
     ▼
Text Preprocessing
  ├─ Lowercase
  ├─ Remove URLs, mentions, hashtags
  ├─ Strip punctuation & digits
  └─ Remove stop words
     │
     ▼
Label Encoding → One-Hot Encoding
     │
     ▼
Tokenization (top 10,000 words)
     │
     ▼
Sequence Padding (max_len = 100)
     │
     ▼
Train / Validation Split (80/20)
     │
     ▼
LSTM Model Training
  ├─ EarlyStopping (patience=5)
  └─ ReduceLROnPlateau (patience=3)
     │
     ▼
Evaluation
  ├─ Accuracy & Loss Curves
  ├─ Classification Report
  └─ Confusion Matrix
     │
     ▼
Save Model (.keras) + Inference
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

Please make sure your code follows PEP 8 style guidelines and includes docstrings for any new functions.

---


---

<p align="center">Made with ❤️ and TensorFlow</p>
