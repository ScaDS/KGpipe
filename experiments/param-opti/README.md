# Parameter Optimization Experiment

This experiment extracts and analyzes configuration parameters from open-source data integration tools using the `kgpipe_parameters` extraction module.

## Paper mock experiments (Quality-Aware Pipelines)

This directory also contains a **self-contained mock** of the experiments described in `Quality_Aware_Pipelines.pdf` (Section 6, “Experimental Evaluation”).

- **What it is**: a small simulation of (a) a pipeline configuration space (implementations + thresholds), (b) a “true” end-to-end quality objective (accuracy/coverage/consistency aggregated), (c) an approximate quality estimator \( \hat{Q} \), and (d) search strategies (Default, Random Search, Quality-Aware Search).
- **What it is not**: it does **not** run KGpipe or reproduce the paper’s numbers. It’s meant as a scaffolding to iterate on the experimental protocol and factor out cleaner subpackages later.

### Run the mock experiments

From `experiments/param-opti`:

```bash
python3 run_qap_mock.py all
python3 run_qap_mock.py exp1   # search effectiveness (Table-2-like)
python3 run_qap_mock.py exp2   # estimation reliability (corr/MAE/top-k)
python3 run_qap_mock.py exp3   # impl-only vs param-only vs joint
```

Outputs are written to `output_qap_mock/` (JSON).

#### “Mock → real” execution mode

The `qap_mock` package can now execute **real KGpipe tasks** (instead of purely simulated formulas) when dependencies are installed.

- **Install dependencies** (from repo root):

```bash
python3 -m pip install -e .
```

- **Enable docker-backed tasks** (PARIS, CoreNLP) for richer pipelines:

```bash
export QAP_MOCK_USE_DOCKER=1
```

Without `QAP_MOCK_USE_DOCKER=1`, `qap_mock` will use non-docker fallbacks where available (e.g., union-only RDF fusion and a lightweight pattern-based IE) so the experiment harness stays runnable.

## Directory Structure

```
param-opti/
├── input/              # Tool definitions
│   ├── paris/
│   │   ├── repo.url    # Git repository URL
│   │   └── cli.txt     # CLI help output (optional)
│   └── corenlp_openie/
│       └── repo.url
├── repos/              # Cloned repositories (auto-populated)
├── output/             # Extraction results (JSON)
├── output_qap_mock/    # Mock paper experiment results (JSON)
├── src/
│   └── param_opti/     # Experiment code
└── run_experiment.py   # Main entry point
```

## Usage

### Run full experiment

```bash
# From kgpipe root (with venv activated)
cd experiments/param-opti
python run_experiment.py
```

### Run for specific tool

```bash
python run_experiment.py --tool paris
python run_experiment.py --tool paris corenlp_openie
```

### Skip repository cloning

```bash
python run_experiment.py --no-clone
```

### Use LLM-based extraction (requires kgpipe_llm)

```bash
python run_experiment.py --use-llm
```

## Adding New Tools

1. Create a folder in `input/` with the tool name
2. Add `repo.url` with the Git repository URL
3. Optionally add `cli.txt` with CLI help output
4. Optionally add `config.json` for additional settings:

```json
{
    "language": "python",
    "main_files": ["src/main.py", "cli.py"]
}
```

## Output Format

Results are saved as JSON files in `output/`:

```json
{
    "tool_name": "paris",
    "timestamp": "2024-...",
    "sources": [
        {
            "source_type": "cli",
            "file_path": "input/paris/cli.txt",
            "parameters_count": 5
        }
    ],
    "parameters": [
        {
            "name": "threshold",
            "native_keys": ["--threshold"],
            "description": "Matching threshold",
            "type_hint": "float",
            "default_value": 0.5,
            "_source": "cli"
        }
    ],
    "summary": {
        "total_parameters": 5,
        "total_sources": 1,
        "total_errors": 0
    }
}
```

A `_summary.json` file is also generated with aggregate statistics.


# Configuration Apsects

1. Task Assignment: Selecting
2. Task Tunning
3. 