"""Helper script to run Streamlit UI."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the Streamlit application."""
    ui_path = Path(__file__).parent.parent / "src" / "rag_evaluator" / "ui" / "streamlit_app.py"

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path)],
        check=True,
    )


if __name__ == "__main__":
    main()
