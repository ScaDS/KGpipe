# Tool Parameter Extraction

An experiment to extract configuration (hyper)parameters from tools that perform data integration tasks and cluster them to show common options.

The extraction and clustering code is under src/kgpipe_parameters
The experiment code using this is under experimets/param-opti

Trace the implementation state under spec/
Reuse its state and extend it for each new feature.
Tack issues under spec/fix_needed.md

# Implementation Requirements
- Implement parameter extractors from
    - docker doc
    - python lib
    - cli help
    - http api doc
    - repo Readme.md/Doc
- The extractors can use LLMs or rules/regex patterns
- Find similar parameters between the single tools implementing a cluster strategy
  - using sentence transformer embeddings
  - prompting llms with preselected terms
- Visualize the clusters

# Success Criteria
- simple tests for each extractors
- a working experiment in experiments/param-opti
- a table with configuration parameters
- A vizualization output

I want you to check missing features and just implement the next feature required now.