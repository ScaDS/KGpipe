from kgpipe_tasks.benchutils import load_matches_filtered, load_matches

selection_tsv ="/home/marvin/project/data/acquisiton/selection_dbp_sameAs_wd_a_film.tsv"

import time

# Measure loading time for load_matches
start_time = time.time()
match_cluster = load_matches(selection_tsv)
end_time = time.time()
print(f"Time to load matches from '{selection_tsv}': {end_time - start_time:.4f} seconds")
