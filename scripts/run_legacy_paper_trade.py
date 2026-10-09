"""Rerun the older paper-trading bot with a different starting bankroll.

    .venv/bin/python scripts/run_legacy_paper_trade.py --cash 1000
    .venv/bin/python scripts/run_legacy_paper_trade.py --cash 1000 --final

The older repo's trained models and its data/daily/{rows,static}.parquet are
required. Pass --data-dir if the daily tables were saved elsewhere. All other
arguments go to the older paper_trade.py. Results are written under this
repo's data/backtests/ so the original $100 trade logs are preserved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY = ROOT.parent / "mtgpricepredictor"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cash", type=float, default=1000.0, help="starting cash per run (default: $1,000)")
    parser.add_argument("--legacy-repo", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--data-dir", type=Path, help="directory containing rows.parquet and static.parquet")
    parser.add_argument("--output-dir", type=Path, help="directory for this run's trade logs")
    args, legacy_args = parser.parse_known_args()
    if args.cash <= 0:
        parser.error("--cash must be positive")
    legacy = args.legacy_repo.resolve()
    daily = (args.data_dir or legacy / "data" / "daily").resolve()
    missing = [p for p in (daily / "rows.parquet", daily / "static.parquet",
                           legacy / "models" / "price_move.joblib",
                           legacy / "models" / "buylist_move.joblib",
                           legacy / "models" / "fair_anchors.parquet") if not p.is_file()]
    if missing:
        parser.error("missing required backtest input(s):\n  " + "\n  ".join(map(str, missing)))

    sys.path.insert(0, str(legacy))
    import fair_value  # noqa: E402
    import paper_trade  # noqa: E402

    fair_value.DAILY = daily
    paper_trade.START_CASH = args.cash
    paper_trade.OUTPUTS = (args.output_dir or ROOT / "data" / "backtests" / f"legacy_cash_{args.cash:g}").resolve()
    paper_trade.OUTPUTS.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(legacy / "paper_trade.py"), *legacy_args]
    paper_trade.main()


if __name__ == "__main__":
    main()
