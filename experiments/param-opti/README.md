# Parameter Optimization Experiment

This experiment extracts and analyzes configuration parameters from open-source data integration tools using the `kgpipe_parameters` extraction module.

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


