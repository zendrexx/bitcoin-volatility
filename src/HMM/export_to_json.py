import os
import pandas as pd


def export_json(df: pd.DataFrame, label: str, output_dir: str = "data/processed") -> str:
   
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"hmm_{label}.json")

    keep_cols = [c for c in ["time", "open", "high", "low", "close", "volume",
                              "log_return", "high_low_range", "rolling_vol",
                              "state", "regime"] if c in df.columns]

    df[keep_cols].to_json(filepath, orient="records", date_format="iso", indent=2)
    return filepath


def export_results_summary(results: list, output_dir: str = "data/processed") -> str:
   
    import json
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "hmm_results_summary.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    return filepath