# FakeScope — AI Fake News Detection

<p align="center">
  <img src="app/static/img/1.jpg" alt="FakeScope Platform Preview" width="100%" />
</p>

Production-ready Flask web app that scores news headlines and articles as **FAKE** or **REAL** using a TF-IDF + scikit-learn pipeline.

## Features

- Polished multi-page UI (Analyze, How it works, API docs)
- JSON API with input validation, rate limiting, and structured errors
- App factory architecture, gunicorn + Docker deployment
- Auto-trains models on first launch if artifacts are missing
- Health and metrics endpoints for ops

## Quick start

```bash
# 1. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (uses dataset/news_dataset.csv or generates one)
python train.py

# 4. Run the development server
python run.py
# → http://127.0.0.1:5000
```
