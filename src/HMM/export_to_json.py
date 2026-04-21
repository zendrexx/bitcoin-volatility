def export_json(df, tf):
    df_out = df[["time", "close", "log_return", "volatility", "state"]]

    df_out.to_json(
        f"data/processed/hmm_{tf}.json",
        orient="records",
        date_format="iso"
    )