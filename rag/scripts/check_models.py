"""Model readiness check (Stage 0, template T1): one tiny call per role.

Usage (from rag/):  python scripts/check_models.py [--judge]
Exit code 1 if any role fails. A 404 "No endpoints found" means the slug is retired.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ConfigError, settings  # noqa: E402
from app.diagnostics import format_t1, run_checks  # noqa: E402


def main() -> int:
    try:
        results = run_checks(settings(), include_judge="--judge" in sys.argv)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}")
        return 2
    for r in results:
        print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['role']:<20} {r['model']:<45} {r['ms']:>6} ms  {r['note']}")
    print("\nT1 (paste into your report):\n" + format_t1(results))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
