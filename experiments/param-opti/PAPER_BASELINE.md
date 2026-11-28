# KGpipe Conf paper baseline

Immutable record of the accepted pipeline-configuration-search experiment
(`experiments/param-opti`). Historical result files are not modified by this
document. No git tag has been created; the commit below is the baseline ref.

## Code

| Item | Value |
|------|--------|
| Baseline commit | `595acb9cf364e41b4c4f6639614074a6f335c5b1` |
| Subject | `exp(params): update README.md for paper` (2026-07-25) |
| Preceding experiment commit | `f7fb821dfc2919fa30473b8a7d591339bd987acb` (`latest paper experiment config and mini fixes`) |
| Merge onto `main` | `3be3ed6d917974cfe7303f37c2a326f1450a7cee` (PR #20) |
| Tree identity | `git diff 595acb9 3be3ed6` is empty. The merge commit and the baseline commit are the same tree. |
| `parameters` branch | Tip is `595acb9`. It is an ancestor of `main`. Already integrated. Do not merge it again. |

Checkout `595acb9` (or `3be3ed6`) to reproduce the paper code. `HEAD` is not that tree.

After the merge, `experiments/param-opti` changed only in `b133147`, which adds the results-repo link to this README. `pyproject.toml`, `uv.lock`, and `.python-version` are unchanged from the baseline through current `main`.

`src/` did change after the merge, in modules this experiment imports:

- `67ca171` adds schema and ontology matching to `BasicTaskCategoryCatalog`
- `a36775e` / PR #23 lazy-loads torch, transformers, and sentence-transformers in core and task modules

A rerun from current `main` is therefore not the paper runtime, even though the experiment package and the lockfile match.

## Dependencies

| Item | Value |
|------|--------|
| Python | 3.12 (`.python-version`, `requires-python >= 3.12`) |
| Project version | `0.7.0` |
| Lockfile | `uv.lock` at `595acb9` (identical on current `main`) |
| `kgcore` | `https://github.com/Vehnem/kgcore.git` @ `62aad8b1937fe613f4b28372d142961fbfe78547` |

## Dataset

Published result JSON records absolute paths on the original machine. The dataset content hash is not stored. The directory name `latest` is a moving label, not a revision.

Root: `/local/d1/datasets/kgpipe-parameters/latest/bench/moviekg_datasets/film_10k/`

| Role | Path under that root |
|------|----------------------|
| Seed KG | `split_0/kg/seed/data.nt` |
| RDF source | `split_1/sources/rdf/data.nt` |
| Text source | `split_1/sources/text/data` (directory) |
| Reference KG | `split_1/kg/reference/data_agg.nt` |
| Ontology | `ontology.ttl` |

The README quick-start uses `data/bench/moviekg/...`. That is a shortened local symlink layout (`experiments/param-opti/data` → `kgpipe-parameters/latest`). The files that were actually scored are the `film_10k` split above: seed from split 0, source and reference from split 1.

This checkout does not contain that dataset (`/local/d1/...` is absent).

## Search settings recorded in the result JSON

Taken from `parameter_search/{rdf,text}-search-results_rank_custom/init_1_budget_20_seed_0/hnr_2-results.json`. The same budget, init, rank, and path pattern hold for the other strategy files in that tree.

| Setting | Value |
|---------|--------|
| Pipeline types | `rdf`, `text` |
| Strategies | `random`, `implementation_aware`, `qgns` (RNS in the results README), `hnr`, `hnr_2`, `bayesian` |
| Not in the published sweep | `llm` |
| Rank aggregation | `custom` (`CUSTOM_AGGREGATION_CONFIG` in `src/kgpipe_search/ranking_conf.py`) |
| Init strategy | `implementation_aware` |
| Init budget / budget | `1` / `20` |
| `y` | `1` |
| `k` | `3` |
| `rho` | `0.2` |
| Bayesian `pool_size` / `beta` | `32` / `0.5` |
| RNG seeds | `0, 1, 2, 3, 4, 5, 6, 7, 42, 1337` |
| `tasks_tmp_scope` | `pipeline` |
| `reuse_existing` | `true` |

`custom` ranking is a harmonic mean of coverage+correctness alignment scores, a harmonic mean of the six consistency scores, then a harmonic mean of those two subgroup scores. Measurements are `EntityAlignmentMetric` and `TripleAlignmentMetric` precision/recall, plus `DisjointDomainMetric`, `DomainMetric`, `RangeMetric`, `DatatypeFormatMetric`, `DatatypeMetric`, and `RelationDirectionMetric` normalized scores.

### HNR-2 stop parameters are not in the artifacts

`min_quality_delta` and `min_iterations_wo_improvement` are not written into the results JSON. At `595acb9` the CLI defaults are `0.01` and `3`. The library defaults on `_run_search` are `0.05` and `2`. `main()` passes the CLI values through, so a driver that omitted the flags would have used `0.01` and `3`. The driver scripts are missing, so an override cannot be ruled out.

## Container images

Tags are floating. No digest is recorded in git or in the result JSON.

| Image | Where |
|-------|--------|
| `kgt/paris:latest` | `dev/tasks/paris_lib.py` (RDF alignment) |
| `kgt/corenlp:latest` | `dev/tasks/corenlp_lip.py` (text extraction) |
| `genie:latest` | `dev/tasks/genie_lib.py`, built from `wrappers/genie/Dockerfile` |
| `local/spotlight` | `docker-compose.yml` service `dbpedia-spotlight` |
| `redis` | compose `cache` (embedding cache when configured) |
| `openlink/virtuoso-opensource-7:latest` | compose `syskg` |

Genie image build: `python:3.8-slim`, unpinned `git clone` of `epfl-dlab/GenIE`, Zenodo records `6139236` (`genie_plm_r.ckpt`, `tries.zip`). Spotlight is not invoked as an image from the task. `spotlight_lib.py` POSTs to `DBPEDIA_ANNOTATE_URL`, default `http://localhost:2222/rest/annotate`, with confidence `0.35`. The paper runs do not record which URL or image digest was live.

`alpine:3.20` and `apache/hadoop:3.3.6` appear only as copy-job defaults in `dev/docker/copy.py`, not as search-space tasks.

Embedding model ids in the search space (not container images): `sentence-transformers/all-MiniLM-L6-v2`, `sentence-transformers/all-mpnet-base-v2`, `intfloat/e5-base-v2`.

## Results revision

Do not rewrite these files. They are the paper numbers.

| Item | Value |
|------|--------|
| Repository | https://github.com/Vehnem/kgpipe-experiment-results |
| Path | `parameter_search/` |
| Commit | `073d914a76d35e8868f27719988b71bdb09c41ab` |
| Date | 2026-07-25T10:47:01Z |
| History | That commit is the only one on `main`. |

Layout: `rdf/`, `text/` (pipeline caches), `rdf_sampled/`, `text_sampled/`, `rdf_reaggregated/`, `text_reaggregated/`, `score_curves/`, and `rdf-search-results_rank_custom/` plus `text-search-results_rank_custom/` with `init_1_budget_20_seed_<s>/` for each seed above. Aggregated tables live next to those seed directories. The results README states RDF expected max ≈ 0.771 and text expected max ≈ 0.380. Seed 0 `hnr_2` RDF `best_score` in the JSON is `0.7712140467593951`.

## Missing reproduction drivers

`experiments/param-opti/README.md` and the results README tell readers to run:

- `full.sh`
- `scripts/rdf_experiments.sh`
- `scripts/text_experiments.sh`
- `scripts/call_plot.sh`
- `scripts/call_execute.sh`
- `scripts/call_text_execute.sh`
- `scripts/call_analyse-offline.sh`
- `scripts/rdf_hnr2_params.sh`
- `scripts/run_multi`

None of these paths exist at `595acb9`, on current `main`, or on any other local or `origin` branch. `git log --all` has no `full.sh` and no `experiments/param-opti/scripts/`. They are not in `kgpipe-experiment-results` and they are not on this machine. There is nothing to restore. The single-experiment CLI in the README, plus the settings table above, is the recoverable interface. Do not invent the shell drivers.

## Branch classification

Decision for this audit: only `parameters` and `demo-fixes` were relevant. `parameters` was already on `main` (PR #20). `demo-fixes` is merged and pushed as `f5a8178`. `text-pipelines` stays for a later cleanup. The other branches were testing.

After that merge was pushed, these `origin` branches were deleted: `parameters`, `demo-fixes`, `demo`, `text-pipelines-squashed`, `cleanup-eval`, `dep-fixes`, `explorer`, `ontologies`, `pdf-source`, `pipekg-explorer`, `refactor-cleanup`, `refactor-eval`, `refactor-syskg`, `regensburg`, `system-kg`, `vldb`.

`origin` still has `main`, `text-pipelines`, `taskdev`, and `theo-ma`.

| Branch | Disposition |
|--------|-------------|
| `parameters` | Deleted on `origin`. Tip `595acb9` remains on `main`. |
| `demo-fixes` | Merged in `f5a8178` and deleted on `origin`. Eight commits for the explorer/system-KG demo: dry-run materialization, `kgp:Pipeline` in the system KG, task metadata, and CLI alignment with kgpipe-explorer. |
| `text-pipelines` | Kept. Clean up later. 89 commits ahead of the pre-merge `main`, diverged at `2bda809`. Tip `f52ebfb` (2026-07-08) is Rebel / text2kgbench work under `experiments/text-pipelines/`. |
| `text-pipelines-squashed` | Deleted on `origin`. Squashed copy of the same line, tip `37c10db` (2026-06-11), short of the July Rebel updates. `text-pipelines` is the remaining copy. |
| `taskdev` | Kept. Testing. Holds `regensburg`, `pdf-source`, and older task WIP (7 commits ahead of the pre-merge `main`). |
| `theo-ma` | Kept. Testing. PyJedAI task update `bff4c2b` (2026-08-06), 2 commits ahead of the pre-merge `main`. |

`mirror/*` still has the older branch names it had before this cleanup, except `text-pipelines-squashed`. No extra paper history lives only on `mirror`.
