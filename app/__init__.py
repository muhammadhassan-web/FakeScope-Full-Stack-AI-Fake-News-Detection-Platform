"""FakeScope Flask application factory."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory

from app.config import BASE_DIR, get_config
from app.extensions import limiter
from app.routes import api_bp, pages_bp
from app.services import predictor as predictor_module
from app.services.predictor import Predictor


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def create_app(config_name: str | None = None) -> Flask:
    cfg = get_config(config_name)
    if getattr(cfg, "ENV", "") == "production" and hasattr(cfg, "validate"):
        cfg.validate()

    configure_logging(cfg.LOG_LEVEL)

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
        static_url_path="/static",
    )
    app.config.from_object(cfg)

    # Ensure project root is importable for preprocess / train
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)

    limiter.init_app(app)

    # Load ML model (skipped in unit tests unless artifacts exist)
    pred = Predictor(Path(app.config["MODEL_DIR"]))
    if not app.config.get("TESTING", False):
        pred.load(auto_train=True)
    else:
        try:
            pred.load(auto_train=False)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.warning("Tests running without trained model artifacts.")
    predictor_module.predictor = pred

    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp)

    # Serve training charts from legacy /static (project root)
    legacy_static = Path(app.config["LEGACY_STATIC_DIR"])

    @app.get("/media/<path:filename>")
    def media(filename: str):
        return send_from_directory(legacy_static, filename)

    @app.errorhandler(404)
    def not_found(err):
        if _wants_json():
            return jsonify({"error": "Not found", "code": "not_found"}), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(429)
    def rate_limited(err):
        if _wants_json():
            return jsonify({"error": "Rate limit exceeded. Try again shortly.", "code": "rate_limited"}), 429
        return render_template("errors/429.html"), 429

    @app.errorhandler(500)
    def server_error(err):
        if _wants_json():
            return jsonify({"error": "Internal server error", "code": "server_error"}), 500
        return render_template("errors/500.html"), 500

    @app.context_processor
    def inject_globals():
        return {
            "app_name": "FakeScope",
            "app_tagline": "AI news authenticity analysis",
        }

    return app


def _wants_json() -> bool:
    from flask import request

    accept = request.accept_mimetypes
    best = accept.best_match(["application/json", "text/html"])
    return (
        best == "application/json"
        and accept[best] >= accept["text/html"]
    ) or request.path.startswith("/predict") or request.path.startswith("/health") or request.path.startswith("/metrics")
