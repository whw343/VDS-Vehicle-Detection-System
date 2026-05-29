"""Start the VDS Flask app in a stable local demo mode."""

from pathlib import Path
import sys
import traceback


BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "flask-server.log"


def main():
    with LOG_PATH.open("a", encoding="utf-8") as log:
        sys.stdout = log
        sys.stderr = log
        try:
            import app

            app.app.run(debug=False, host="127.0.0.1", port=5000)
        except Exception:
            traceback.print_exc(file=log)
            raise


if __name__ == "__main__":
    main()
