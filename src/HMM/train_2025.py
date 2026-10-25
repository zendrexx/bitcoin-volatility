"""Train and store the 2025 HMM bundles used for live inference.

Examples:
    python src/HMM/train_2025.py --coin btc --tf 15m
    python src/HMM/train_2025.py --coin all --tf all
"""

import argparse
from threadpoolctl import threadpool_limits

from server import SUPPORTED_COINS, TIMEFRAMES, MODEL_DIR, train_2025_models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coin", choices=[*SUPPORTED_COINS, "all"], default="all")
    parser.add_argument("--tf", choices=[*TIMEFRAMES, "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Replace saved models")
    args = parser.parse_args()

    coins = SUPPORTED_COINS if args.coin == "all" else [args.coin]
    timeframes = TIMEFRAMES if args.tf == "all" else [args.tf]
    failed = False
    for coin in coins:
        for tf in timeframes:
            print(f"\nTraining {coin.upper()} {tf} from 2025 candles...")
            try:
                with threadpool_limits(limits=1):
                    statuses = train_2025_models(coin, tf, force=args.force)
            except (FileNotFoundError, ValueError) as exc:
                print(f"  Failed: {exc}")
                failed = True
                continue
            for status in statuses:
                print(f"  Model {status['model']}: {status['status']}"
                      + (f" ({status['error']})" if "error" in status else ""))
                failed |= status["status"] == "error"
    print(f"\nSaved model directory: {MODEL_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
