"""Long-lived process for the desktop app: pay the torch import and model warm-up once, then run jobs.

Prints "@@ready" when warmed up. Each stdin line is a job, {"command": "run" | "corrections", "argv": [...]},
handled by the same main() as the CLI; its output is followed by "@@done <exit code>".
"""
import json
import sys
import traceback

import numpy as np

import run
from src import corrections
from src.slot_inference import SlotInspector

COMMANDS = {"run": run.main, "corrections": corrections.main}


def warm_up() -> None:
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    for model_dir in run.MODEL_DIRS.values():
        if model_dir.is_dir():
            SlotInspector(model_dir).classify([crop])


def run_job(line: str) -> int:
    try:
        job = json.loads(line)
        COMMANDS[job["command"]](job["argv"])
    except SystemExit as exit_:
        if exit_.code is None or isinstance(exit_.code, int):
            return exit_.code or 0
        print(exit_.code)  # sys.exit("message") prints the message, as the CLI would
        return 1
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def main() -> None:
    # Live per-image progress through the pipe, and paths/lot names intact whatever the Windows code page.
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr = sys.stdout  # one stream keeps log lines in order with the @@done marker
    warm_up()
    print("@@ready", flush=True)
    for line in sys.stdin:
        if line.strip():
            code = run_job(line)
            print(f"@@done {code}", flush=True)


if __name__ == "__main__":
    main()
