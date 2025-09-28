# kg-geneval
Documentation and experiment code for incremental KG generation and evaluation.

Requirements:
> wget, tar, git \
> docker \
> micromamba

# Run Experiment

```bash
make reproduce

# only tesgs
make reproduce-tests-only
```

# Configuration
There are the following configuration files `dot.env` or `.env` and `pipeline.conf`
You can change and adapt the configuration for you own purpose.

## Input Structure
```
inc-moviekg-bench/
├── results
├── sources
│   ├── aggregated
│   ├── seed.nt
│   └── splits
└── testing
```

## Output Structure

```
benchmark_name/
├── results
│   ├── pipeline_name_1
│   │   ├── plan.json
│   │   ├── stage_1
│   │   │   ├── eval-report.json
│   │   │   ├── exec-plan.json
│   │   │   ├── exec-report.json
│   │   │   └── tmp
│   │   │       └── task_name_1_output.format
│   │   └── stage_2
│   │       └── ...
│   └── pipeline_name_2
│       └── ...
├── sources
└── testing
```

# Licence
TODO

# Backlog
- Reuse SSP results as stage_1 of the MSPs