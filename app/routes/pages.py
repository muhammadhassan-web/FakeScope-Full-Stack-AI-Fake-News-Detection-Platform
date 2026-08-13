"""Page routes."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, render_template

from app.services.predictor import get_predictor

pages_bp = Blueprint("pages", __name__)

EXAMPLES = [
    {
        "label": "real",
        "title": "MIT antibiotic discovery",
        "text": (
            "Scientists at MIT have identified a novel antibiotic compound "
            "using machine learning, offering hope against MRSA and other "
            "resistant strains in peer-reviewed trials."
        ),
    },
    {
        "label": "real",
        "title": "Federal Reserve rate hike",
        "text": (
            "The Federal Reserve increased benchmark rates for the third "
            "consecutive quarter, citing persistent consumer price inflation "
            "at 3.8% annually per official CPI data."
        ),
    },
    {
        "label": "fake",
        "title": "Vaccine microchips claim",
        "text": (
            "BREAKING: vaccines contain mind-control microchips activated by "
            "5G. Brave doctors confirm nano-chips sync with towers to track "
            "your thoughts in real time. Share before they delete this!"
        ),
    },
    {
        "label": "fake",
        "title": "Secret cancer cure",
        "text": (
            "Secret cure for cancer suppressed by Big Pharma for 50 years. "
            "A natural compound in apricot seeds was proven in 1972 to cure "
            "ALL cancers but the FDA was paid off to hide this $3 remedy!"
        ),
    },
]


def _has_charts() -> bool:
    legacy = Path(current_app.config["LEGACY_STATIC_DIR"])
    needed = ("comparison_chart.png", "confusion_matrices.png", "performance_overview.png")
    return all((legacy / name).exists() for name in needed)


@pages_bp.get("/")
def index():
    try:
        pred = get_predictor()
        model_name = pred.model_name
        metrics = pred.metrics
    except Exception:
        model_name = "Unavailable"
        metrics = {}

    live_factcheck = bool(current_app.config["TAVILY_API_KEY"] and current_app.config["GROQ_API_KEY"])

    return render_template(
        "index.html",
        model_name=model_name,
        metrics=metrics,
        examples=EXAMPLES,
        has_charts=_has_charts(),
        min_length=current_app.config["MIN_TEXT_LENGTH"],
        max_length=current_app.config["MAX_TEXT_LENGTH"],
        live_factcheck=live_factcheck,
    )


@pages_bp.get("/about")
def about():
    try:
        pred = get_predictor()
        model_name = pred.model_name
        metrics = pred.metrics
    except Exception:
        model_name = "Unavailable"
        metrics = {}

    return render_template(
        "about.html",
        model_name=model_name,
        metrics=metrics,
        has_charts=_has_charts(),
    )


@pages_bp.get("/docs")
def docs():
    return render_template(
        "docs.html",
        min_length=current_app.config["MIN_TEXT_LENGTH"],
        max_length=current_app.config["MAX_TEXT_LENGTH"],
    )
