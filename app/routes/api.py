"""JSON API routes."""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from app.extensions import limiter
from app.services.factcheck import FactCheckServiceError, FactCheckUnavailableError, verify_claim
from app.services.predictor import ModelNotReadyError, get_predictor
from app.services.validation import validate_news_text

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)


@api_bp.post("/predict")
@limiter.limit(lambda: current_app.config["RATELIMIT_PREDICT"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Expected JSON body with a 'text' field.", "code": "invalid_json"}), 400

    text, err = validate_news_text(
        payload.get("text"),
        min_length=current_app.config["MIN_TEXT_LENGTH"],
        max_length=current_app.config["MAX_TEXT_LENGTH"],
    )
    if err:
        return jsonify({"error": err.message, "code": err.code}), 400

    try:
        result = get_predictor().predict(text)
        return jsonify(result.to_dict())
    except ValueError as exc:
        return jsonify({"error": str(exc), "code": "unusable_text"}), 422
    except ModelNotReadyError as exc:
        logger.exception("Model not ready")
        return jsonify({"error": str(exc), "code": "model_unavailable"}), 503
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Internal prediction error.", "code": "prediction_failed"}), 500


@api_bp.post("/verify")
@limiter.limit(lambda: current_app.config["RATELIMIT_VERIFY"])
def verify():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Expected JSON body with a 'text' field.", "code": "invalid_json"}), 400

    text, err = validate_news_text(
        payload.get("text"),
        min_length=current_app.config["MIN_TEXT_LENGTH"],
        max_length=current_app.config["MAX_TEXT_LENGTH"],
    )
    if err:
        return jsonify({"error": err.message, "code": err.code}), 400

    try:
        result = verify_claim(
            text,
            tavily_api_key=current_app.config["TAVILY_API_KEY"],
            groq_api_key=current_app.config["GROQ_API_KEY"],
            groq_base_url=current_app.config["GROQ_BASE_URL"],
            groq_model=current_app.config["GROQ_MODEL"],
            max_sources=current_app.config["FACTCHECK_MAX_SOURCES"],
        )
        return jsonify(result.to_dict())
    except FactCheckUnavailableError as exc:
        return jsonify({"error": str(exc), "code": "factcheck_unavailable"}), 503
    except FactCheckServiceError as exc:
        logger.exception("Fact-check upstream failure")
        return jsonify({"error": str(exc), "code": "factcheck_failed"}), 502
    except Exception:
        logger.exception("Fact-check failed")
        return jsonify({"error": "Internal fact-check error.", "code": "factcheck_failed"}), 500


@api_bp.get("/health")
@limiter.exempt
def health():
    try:
        pred = get_predictor()
        ready = pred.ready
        model = pred.model_name
    except Exception:
        ready = False
        model = None

    status = "ok" if ready else "degraded"
    code = 200 if ready else 503
    return jsonify({
        "status": status,
        "model": model,
        "ready": ready,
        "live_factcheck": bool(current_app.config["TAVILY_API_KEY"] and current_app.config["GROQ_API_KEY"]),
    }), code


@api_bp.get("/metrics")
def metrics():
    try:
        pred = get_predictor()
        return jsonify({"model": pred.model_name, "metrics": pred.metrics})
    except ModelNotReadyError as exc:
        return jsonify({"error": str(exc), "code": "model_unavailable"}), 503
