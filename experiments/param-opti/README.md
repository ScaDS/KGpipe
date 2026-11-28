# Pipeline configuration search

Search over KG integration pipeline configs (task selection + parameters) to maximize evaluation quality against a reference KG. Supports **RDF** (graph alignment / fusion) and **text** (IE → linking → RDF → fusion) pipelines on the MovieKG benchmark.

Experiment results can be found at https://github.com/Vehnem/kgpipe-experiment-results/tree/main/parameter_search. A local copy of the reports is `data/kgpipe-experiment-results/parameter_search`. A search candidate is compiled to a `kgpipe.pipeline/v1` catalog before it is built. Sampling and category coverage come from `kgpipe.generation.space`; the `{task_key: {category, parameter: [values]}}` tables in `definitions.py` are the paper's slot and domain data, checked against each task's declared parameters.

Result files are named by the SHA-256 of the canonical `{task_keys, profiles}` JSON. That digest is unchanged by the `kgpipe.pipeline/v1` plan, whose `fingerprint` is a separate identity. `src/kgpipe_search/test/test_legacy_snapshots.py` checks the digest for `rdf_sampled`, `text_sampled`, `rdf`, and `text`. `src/kgpipe_search/test/test_candidate_spec.py` rebuilds the sampled RDF and text runs and checks task order, port refs, parameter values, and the fingerprint. Neither test executes the tasks. The snapshot format is documented in `docs/parameters.md`.

The accepted-paper checkout, dataset split, ranking settings, image tags, and results commit are recorded in [PAPER_BASELINE.md](PAPER_BASELINE.md). `scripts/` and `full.sh` were never committed; the commands below are the intended interface, not files in this tree.

## Layout

| Path | Role |
|------|------|
| `src/experiment.py` | Live search: propose configs → run pipeline → evaluate → write results |
| `src/execute.py` | Run/evaluate fixed config fixtures (sampled or exhaustive) |
| `src/analyse.py` | Offline search simulation on a cached `results.json` |
| `src/plot_search_evolution.py` | Per-seed evolution plots/tables |
| `src/plot_search_evolution_aggregate.py` | Mean ± band across RNG seeds |
| `src/kgpipe_search/` | Search space, strategies, ranking, evaluation |
| `scripts/` | Reproducible experiment drivers |
| `data/` | Symlink to MovieKG bench data (`kgpipe-parameters/latest`) |
| `runs/` | Pipeline artifacts + search result summaries |

## Search strategies

| Flag | Behavior |
|------|----------|
| `random` | Uniform sample over exhaustive valid configs |
| `implementation_aware` | Systematic task-combo coverage, random params |
| `qgns` | Restricted neighborhood search (RNS in plots) |
| `hnr` / `hnr_2` | Hierarchical neighborhood refinement |
| `bayesian` | Surrogate + acquisition over a candidate pool |
| `llm` | LLM-proposed configs (needs `KGPipe_SEARCH_LLM_*` env vars) |

Objective score comes from `--rank-aggregation` (`default` | `flat_hmean` | `custom`), applied to cached metric measurements in `.eval.json`.

## Quick start

From the **repo root**, with `.venv` and `experiments/param-opti/data` pointing at the bench dataset:

```bash
cd experiments/param-opti

# One seed, all strategies (RDF or text)
bash scripts/rdf_experiments.sh 0
bash scripts/text_experiments.sh 0

# Multi-seed sweep (seeds: 0 42 1337 1–7)
bash full.sh

# Per-seed + aggregated plots
bash scripts/call_plot.sh 0
```

Results land under:

```
runs/{rdf,text}/                                          # pipeline caches (by config hash)
runs/{rdf,text}-search-results_rank_<agg>/init_<i>_budget_<b>_seed_<s>/
  {random,implementation-aware,qgns,hnr,hnr_2,bayesian}-results.json
```

## Single experiment

```bash
export PYTHONPATH=src:<repo>/src
python src/experiment.py \
  --seed data/bench/moviekg/split_0/kg/seed/data.nt \
  --source data/bench/moviekg/split_1/sources/rdf/data.nt \
  --reference data/bench/moviekg/split_1/kg/reference/data_agg.nt \
  --ontology data/bench/moviekg/ontology.ttl \
  --pipeline-type rdf \
  --strategy hnr_2 \
  --budget 20 --init-budget 1 \
  --init-strategy implementation_aware \
  --rank-aggregation custom \
  --rng-seed 0 \
  --output-dir runs/rdf \
  --results runs/rdf-search-results_rank_custom/init_1_budget_20_seed_0/hnr_2-results.json
```

For text, use `--pipeline-type text` and `--source data/bench/moviekg/split_1/sources/text/data/`.

## Related scripts

- `scripts/call_execute.sh` / `call_text_execute.sh` — fixture execution via `execute.py`
- `scripts/call_analyse-offline.sh` — replay strategies on a cached results file
- `scripts/rdf_hnr2_params.sh` / `run_multi` — HNR-2 hyperparameter sweeps

## Lint

```bash
.venv/bin/ruff check --fix experiments/param-opti/src/kgpipe_search
```

`kgpipe_search.dev` is intended to move into the core KGpipe API.
