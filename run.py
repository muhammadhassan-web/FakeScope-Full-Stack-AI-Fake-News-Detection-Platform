"""Development entrypoint: python run.py"""

import os

from app import create_app

app = create_app(os.environ.get("FLASK_ENV", "development"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"\nFakeScope running — http://127.0.0.1:{port}\n")
    app.run(host=host, port=port, debug=app.config.get("DEBUG", False))
