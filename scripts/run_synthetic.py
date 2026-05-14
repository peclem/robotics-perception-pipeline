"""
Run the pipeline on the synthetic camera — no hardware required.
Useful for validating the full stack end-to-end in CI.

Usage:
    python scripts/run_synthetic.py
    RERUN_ENABLED=false python scripts/run_synthetic.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys

if __name__ == "__main__":
    if "--source" not in sys.argv:
        sys.argv.extend(["--source", "synthetic"])
    from launch import main
    main()
