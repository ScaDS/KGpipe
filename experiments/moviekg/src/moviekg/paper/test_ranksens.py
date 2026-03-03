import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# =========================
# 1) Data
# =========================
data = [
    ("T_C", 0.824, 0.367, 0.332),
    ("R_A", 0.996, 0.993, 0.994),
    ("TJR", 0.980, 0.980, 0.793),
    ("RJT", 0.981, 0.967, 0.849),
    ("TRJ", 0.980, 0.967, 0.808),
    ("JRT", 0.982, 0.980, 0.838),
    ("J_A", 0.938, 0.976, 0.988),
    ("T_B", 0.893, 0.555, 0.580),
    ("J_B", 0.968, 0.961, 0.788),
    ("JTR", 0.981, 0.980, 0.806),
    ("J_C", 0.993, 0.751, 0.851),
    ("R_B", 0.993, 0.982, 0.962),
    ("RTJ", 0.979, 0.967, 0.845),
    ("R_C", 0.996, 0.984, 0.993),
    ("T_A", 0.986, 0.526, 0.590),
]
df = pd.DataFrame(data, columns=["pipeline", "semantic", "correctness", "coverage"])

# =========================
# 2) Define cohorts
# =========================
# Single-source pipelines: "R_A", "J_B", "T_C", etc.
single_re = re.compile(r"^[RJT]_[A-Z]$")

df["is_single"] = df["pipeline"].apply(lambda s: bool(single_re.match(s)))
df["source_type"] = df["pipeline"].apply(lambda s: s[0])  # 'R', 'J', 'T'

single_df = df[df["is_single"]].copy()
multi_df = df[~df["is_single"]].copy()  # e.g., "TJR", "RJT", ...

# Cohort dict: RDF-only, JSON-only, TEXT-only, and Multi-source
cohorts = {
    "RDF-only (R_*)": single_df[single_df["source_type"] == "R"].copy(),
    "JSON-only (J_*)": single_df[single_df["source_type"] == "J"].copy(),
    "Text-only (T_*)": single_df[single_df["source_type"] == "T"].copy(),
    "Multi-source (no underscore)": multi_df.copy(),
}

# =========================
# 3) Weight grid on simplex
# =========================
# Weights are (w_sem, w_cor, w_cov) with w_sum=1 and w_i>=0
STEP = 0.05  # set to 0.1 for fewer points
vals = np.round(np.arange(0, 1 + 1e-9, STEP), 10)

weights = []
for w in product(vals, repeat=3):
    if abs(sum(w) - 1.0) < 1e-9:
        weights.append(w)
weights = np.array(weights)  # (N, 3)
print(f"Weight grid: step={STEP}, N={len(weights)} points")

# =========================
# 4) Sensitivity computation
# =========================
def sensitivity_summary(cohort_df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """
    Returns per-pipeline:
      - wins: how many weight points where it ranks #1
      - win_fraction
      - avg_rank
      - avg_score (mean across weights)
    """
    if cohort_df.empty:
        return pd.DataFrame()

    M = cohort_df[["semantic", "correctness", "coverage"]].to_numpy()  # (m,3)
    scores = weights @ M.T  # (N,m)

    # winner counts
    winner_idx = np.argmax(scores, axis=1)
    winners = cohort_df["pipeline"].iloc[winner_idx].to_numpy()
    win_counts = pd.Series(winners).value_counts().reindex(cohort_df["pipeline"]).fillna(0).astype(int)

    # rank matrix: rank 1 = best
    order = scores.argsort(axis=1)[:, ::-1]
    rank_matrix = np.empty_like(order)
    for i in range(order.shape[0]):
        rank_matrix[i, order[i]] = np.arange(1, M.shape[0] + 1)

    summary = pd.DataFrame({
        "wins": win_counts.values,
        "win_fraction": (win_counts.values / len(weights)),
        "avg_rank": rank_matrix.mean(axis=0),
        "avg_score": scores.mean(axis=0),
    }, index=cohort_df["pipeline"].values)

    summary = summary.sort_values(["win_fraction", "avg_rank"], ascending=[False, True])
    return summary

all_summaries = {name: sensitivity_summary(cdf, weights) for name, cdf in cohorts.items()}

# Print summaries
for name, summ in all_summaries.items():
    print("\n" + "=" * 80)
    print(name)
    if summ.empty:
        print("(empty cohort)")
    else:
        print(summ)

# =========================
# 5) Plots (VLDB-friendly)
# =========================
# A) Win-fraction bars for each cohort
# for name, summ in all_summaries.items():
#     if summ.empty:
#         continue
#     plt.figure(figsize=(9, 3.8))
#     plt.bar(summ.index, summ["win_fraction"].values)
#     plt.xticks(rotation=45, ha="right")
#     plt.ylabel("Win fraction (#1 over weight grid)")
#     plt.title(f"{name} — winner sensitivity (step={STEP})")
#     plt.tight_layout()
#     plt.show()

# B) Average-rank bars for each cohort
# for name, summ in all_summaries.items():
#     if summ.empty:
#         continue
#     plt.figure(figsize=(9, 3.8))
#     plt.bar(summ.index, summ["avg_rank"].values)
#     plt.xticks(rotation=45, ha="right")
#     plt.ylabel("Average rank (lower is better)")
#     plt.title(f"{name} — average rank over weight grid")
#     plt.tight_layout()
#     plt.show()

# =========================
# 6) Optional: a compact “paper table” per cohort
# =========================
paper_tables = {}
for name, summ in all_summaries.items():
    if summ.empty:
        continue
    paper_tables[name] = summ[["win_fraction", "avg_rank"]].copy()

print("\n" + "=" * 80)
print("Compact paper tables (win_fraction, avg_rank):")
for name, t in paper_tables.items():
    print("\n---", name, "---")
    print(t)

# =========================
# 7) Optional: export to CSV (uncomment if you want files)
# =========================
# for name, summ in all_summaries.items():
#     if summ.empty:
#         continue
#     safe_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
#     summ.to_csv(f"sensitivity_{safe_name}.csv")
# print("Wrote CSV files.")