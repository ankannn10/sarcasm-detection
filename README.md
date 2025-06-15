# InsightSphere

**Sarcasm-Aware Emotion and Engagement Analyzer for YouTube Videos**

InsightSphere is a Flask-based web application that analyzes YouTube comments in context of their transcript to detect emotion and sarcasm. It combines NLP, deep learning, and interactive dashboards to help content creators and marketers understand public reaction more deeply.

---

## 🔧 Features

* Scrapes YouTube video metadata, transcript, and top comments
* Cleans and preprocesses raw text with slang normalization and emoji removal
* Segments transcripts and matches comments with their most relevant chunk (using Sentence-BERT)
* Predicts top-3 emotions with confidence scores using a DistilRoBERTa-based classifier
* Detects sarcasm using a cross-attention RoBERTa-based model
* Displays visual analytics (pie charts, bar graphs, sentiment tables)
* Supports user authentication (signup/login)

---

## 🗂️ Project Structure

```
├── app.py                      # Flask app with routes and rendering logic
├── auth.py                     # User authentication using SQLite
├── base.html                   # Common HTML layout
├── clean.py                    # Text cleaning, preprocessing, and transcript chunking
├── driver.py                   # Main orchestrator script for the full pipeline
├── inference.py                # Emotion prediction module
├── sarcasm.py                  # Cross-attention sarcasm detection model
├── merge.py                    # Merges sarcasm and emotion CSV outputs
├── pairing.py                  # Comment-transcript pairing using Sentence-BERT
├── scraper.py                  # YouTube data scraper (Selenium + youtube-comment-downloader)
├── templates/
│   ├── index.html              # Input form for YouTube links
│   ├── login.html              # Login form
│   ├── signup.html             # Signup form
│   └── results.html            # Results and charts page
├── static/
│   └── styles.css              # Custom CSS styling
├── output/                     # Intermediate and final output CSVs
│   ├── cleaned_pairs.csv
│   ├── relevant_chunks.csv
│   ├── emotion_predictions.csv
│   └── dependent_emotion_classification.csv
```

---

## 🛠️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # for macOS/Linux
# OR
.venv\Scripts\activate     # for Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Navigate to `http://localhost:5000` in your browser.

---

## 🔮 Model Details

* **Emotion Model**: Fine-tuned DistilRoBERTa with attention pooling
* **Sarcasm Model**: Custom RoBERTa-based cross-attention architecture that considers both comment and transcript context
* **Pairing**: Sentence-BERT cosine similarity for finding most relevant transcript chunk per comment

---

## 💚 License

This project was developed as a final year academic submission. Free for educational or non-commercial use. Contact the author for permission to reuse or extend in commercial settings.

---

## 🚀 Future Enhancements

* Interactive heatmaps for sarcasm distribution across transcript timeline
* REST API for integration into external dashboards or tools
* Upload CSV and bulk inference support
# sarcasm-detection
