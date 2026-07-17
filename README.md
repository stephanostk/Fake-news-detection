# 📰 Fake News Detection System

A full-stack machine learning application that classifies news headlines/articles as **Fake** or **Real**, with a confidence score, built entirely for local execution — no cloud, no containers, no DevOps overhead.

---

## Overview

This project detects potentially fake news using a Natural Language Processing (NLP) pipeline built on TF-IDF vectorization and Logistic Regression. It's a complete, working example of the ML lifecycle: dataset → trained model → API → interactive UI → persistent logging.

**Tech stack:** Python · scikit-learn · FastAPI · Streamlit · MySQL

---

## Architecture

```
User Input (Streamlit UI)
        │
        ▼
FastAPI Backend  ──►  ML Model (TF-IDF + Logistic Regression)
        │
        ▼
MySQL Database (logs every prediction)
```

| Layer | Technology | Runs At |
|---|---|---|
| Frontend | Streamlit | `localhost:8501` |
| Backend | FastAPI | `localhost:8000` |
| ML Model | scikit-learn (TF-IDF + Logistic Regression) | In-process with backend |
| Database | MySQL | `localhost:3306` |

---

## Project Structure

```
fake-news-detector/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── ml/
│   ├── train_model.ipynb
│   ├── model.pkl
│   └── vectorizer.pkl
├── backend/
│   └── main.py
├── frontend/
│   └── app.py
├── venv/
└── README.md
```

---

## Setup Instructions

### 1. Clone / download the project and set up the environment

```bash
cd fake-news-detector
python -m venv venv
```

**Activate the virtual environment:**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

> **Windows PowerShell users:** if you hit a script execution error, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 2. Install dependencies

```bash
pip install pandas scikit-learn numpy joblib jupyter fastapi uvicorn streamlit sqlalchemy pymysql requests
```

### 3. Download the dataset

Get **Fake.csv** and **True.csv** from the [ISOT Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place both files in the `data/` folder.

### 4. Set up MySQL

```sql
CREATE DATABASE fake_news_db;

USE fake_news_db;

CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    input_text TEXT NOT NULL,
    label VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 5. Train the model

Open `ml/train_model.ipynb` in Cursor/VS Code and run all cells. This will produce `model.pkl` and `vectorizer.pkl` inside the `ml/` folder.

### 6. Configure the backend

In `backend/main.py`, set your MySQL credentials:

```python
DB_USER = "root"
DB_PASSWORD = "your_mysql_password"
DB_HOST = "localhost"
DB_NAME = "fake_news_db"
```

---

## Running the Project

Two terminals are needed — one for the backend, one for the frontend.

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --reload
```
API runs at `http://127.0.0.1:8000` · Interactive docs at `http://127.0.0.1:8000/docs`

**Terminal 2 — Frontend:**
```bash
cd frontend
streamlit run app.py
```
UI opens automatically at `http://localhost:8501`

---

## Usage

1. Open the Streamlit UI
2. Paste a news headline or article into the text box
3. Click **Check Credibility**
4. View the predicted label (Fake/Real) and confidence score
5. Every prediction is automatically logged to MySQL (`predictions` table)

---

## Model Performance

Baseline model: **TF-IDF + Logistic Regression**

| Metric | Score |
|---|---|
| Accuracy | ~99% |
| Precision (Fake / Real) | 0.99 / 0.99 |
| Recall (Fake / Real) | 0.99 / 0.99 |

> **Note:** This accuracy is likely inflated by a known characteristic of the ISOT dataset — all "Real" articles come from Reuters specifically, while "Fake" articles come from a variety of other sources. As a result, the model may be partly learning source-specific writing style (e.g., Reuters formatting conventions) rather than purely factual credibility. Manual testing on headlines written outside the dataset's style is recommended before treating these numbers as a general-purpose benchmark.

---

## Known Limitations

- The model's confidence should not be treated as a definitive credibility verdict — it reflects learned statistical patterns, not fact-checking
- Performance may degrade on text with different structure/style than the training data (e.g., social media posts, non-Reuters formatted news)
- No authentication or rate-limiting on the API — intended for local/personal use only

---

## Future Enhancements

- Upgrade model to a fine-tuned DistilBERT transformer for improved generalization
- Add a history/analytics dashboard to the frontend
- Source credibility cross-referencing
- Multilingual support
- Browser extension for real-time flagging

---

## Tech Stack Summary

- **Language:** Python 3
- **ML:** scikit-learn, pandas, joblib
- **Backend:** FastAPI, Uvicorn, SQLAlchemy, PyMySQL
- **Frontend:** Streamlit
- **Database:** MySQL
- **Dataset:** [ISOT Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## Author

**Utkarsh**
B.Tech Computer Engineering (Data Science)
Dr. D. Y. Patil Institute of Technology
