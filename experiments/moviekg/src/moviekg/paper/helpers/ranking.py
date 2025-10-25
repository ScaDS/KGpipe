import pandas as pd
from moviekg.config import OUTPUT_ROOT

def _rank_and_save(weights: dict, outfile_stem: str, df: pd.DataFrame, round_digits: int = 3) -> None:
    """
    Compute weighted 'combined' score and save a TSV sorted by 'combined'.
    Uses the same behavior as your original functions (round to 3, keep default index in CSV).
    """
    cols = ["size", "semantic", "reference", "efficiency"]
    # Ensure we only use known columns; fill missing weights with 0.0
    w = pd.Series(weights).reindex(cols, fill_value=0.0)

    # Compute combined score
    df = df[["pipeline"] + cols].copy()
    df["combined"] = (df[cols] * w).sum(axis=1).round(round_digits)

    # Sort & save (keep default index=True to match original behavior)
    out = df[["pipeline", "combined"]].sort_values(by="combined", ascending=False)
    out.to_csv(OUTPUT_ROOT / f"paper/{outfile_stem}.csv", sep="\t")
