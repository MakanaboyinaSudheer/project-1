import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.app import discover_runs


def test_discover_runs_handles_missing_directory(tmp_path: Path) -> None:
    missing_runs = tmp_path / "no-runs"

    runs = discover_runs(missing_runs)

    assert runs == []
