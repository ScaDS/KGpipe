from moviekg.evaluation.test_eval_refactor import KgBenchData

"""
for every verified_seed remove in the bench data remove the seed entities and store as verified_entities_no_seed.csv
"""

import pandas as pd
from pathlib import Path

bench_data = KgBenchData.from_path(Path("/home/marvin/phd/data/moviekg/datasets/film_1k"))

for i in range(1, 4):
    seed = bench_data.dataset.splits[f"split_{0}"].kg_reference.meta.entities.file
    current = bench_data.dataset.splits[f"split_{i}"].kg_reference.meta.entities.file
    current_path = bench_data.dataset.splits[f"split_{i}"].kg_reference.meta.entities.file
    current_new = current_path.with_name(f"{current_path.stem}_no_seed{current_path.suffix}")

    # remove all lines from current that are in seed and save to new file
    with open(current, "r") as f:
        current_lines = f.readlines()
    with open(seed, "r") as f:
        seed_lines = f.readlines()
    with open(current_new, "w") as f:
        if not current_lines:
            continue

        # Preserve header (assumes first line is the CSV header)
        f.write(current_lines[0])

        seed_set = set(seed_lines[1:] if seed_lines else [])
        for line in current_lines[1:]:
            if line not in seed_set:
                f.write(line)
