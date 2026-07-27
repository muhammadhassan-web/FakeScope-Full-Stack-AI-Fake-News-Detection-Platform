# FakeScope — AI Fake News Detection

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

Copy `.env.example` to `.env` and set a strong `SECRET_KEY` before production use.

## Production

### Gunicorn

```bash
set FLASK_ENV=production
set SECRET_KEY=your-long-random-secret
gunicorn -c gunicorn.conf.py wsgi:app
```

### Docker

```bash
docker compose up --build
# → http://127.0.0.1:5000
```

The image trains models at build time so containers start ready to serve.

## API

### `POST /predict`

```bash
curl -s -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Federal Reserve raises interest rates amid inflation concerns reported by CPI data.\"}"
```

Response fields: `label`, `confidence`, `fake_prob`, `real_prob`, `cleaned_tokens`, `model`, `verdict`, `explanation`.

### `GET /health`

Returns `{ "status": "ok", "model": "…", "ready": true }`.

### `GET /metrics`

Returns training metrics for all compared models.

## Project layout

```
fake-news-detector/
├── app/                    # Flask application package
│   ├── __init__.py         # App factory
│   ├── config.py
│   ├── extensions.py
│   ├── routes/             # Pages + API blueprints
│   ├── services/           # Predictor + validation
│   ├── static/             # CSS, JS, favicon
│   └── templates/          # Jinja templates
├── dataset/                # CSV + generator
├── models/                 # Trained artifacts (generated)
├── static/                 # Training charts (generated)
├── tests/
├── train.py
├── preprocess.py
├── run.py                  # Dev server
├── wsgi.py                 # Production WSGI
├── Dockerfile
└── docker-compose.yml
```

## How the model works

1. **Preprocess** — lowercase, strip URLs/noise, stopword filter, light stemming  
2. **TF-IDF** — up to 15k features, unigrams + bigrams  
3. **Train** — Logistic Regression, Naive Bayes, Random Forest; best F1 is served  
4. **Predict** — class label + calibrated-style probabilities and a short explanation  

For stronger real-world accuracy, replace `dataset/news_dataset.csv` with a larger corpus such as the [Kaggle Fake and Real News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), then re-run `python train.py`.

## Tests

```bash
pytest -q
```

## Disclaimer

FakeScope provides **linguistic authenticity signals**, not verified fact-checks. Always confirm critical claims with primary sources and professional fact-checkers.
