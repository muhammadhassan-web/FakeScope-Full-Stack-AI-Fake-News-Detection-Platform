"""JSON API routes."""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from app.extensions import limiter
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
    return jsonify({"status": status, "model": model, "ready": ready}), code


@api_bp.get("/metrics")
def metrics():
    try:
        pred = get_predictor()
        return jsonify({"model": pred.model_name, "metrics": pred.metrics})
    except ModelNotReadyError as exc:
        return jsonify({"error": str(exc), "code": "model_unavailable"}), 503
