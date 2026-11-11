import os
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "images" / "work" / "runtime"))
sys.path.insert(0, str(TASK_ROOT / "images" / "judge"))
sys.path.insert(0, str(TASK_ROOT / "scripts"))

# The tests cover the Judge too: register the hidden suite's other workloads and metrics (the Work
# runtime alone knows only the public ones; test_protocol checks that).
from kvzip_eval import protocol as _P  # noqa: E402

PUBLIC_WORKLOADS = dict(_P.WORKLOADS)
from kvzip_judge import hidden_workloads as _hidden  # noqa: E402

_hidden.register()


def suite_dir() -> Path | None:
    """Generated suites (scripts/build_suites.sh); KVZIP_SUITE_DIR or the default asset location."""
    if os.environ.get("KVZIP_SUITE_DIR"):
        return Path(os.environ["KVZIP_SUITE_DIR"])
    root = os.environ.get("INFRATHON_ASSETS_ROOT", str(Path.home() / ".cache/infrathon/assets"))
    return Path(root) / TASK_ROOT.name / "generated"
