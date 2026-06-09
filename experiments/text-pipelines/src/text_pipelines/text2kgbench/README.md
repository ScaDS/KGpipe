## Usage

### 0. Convert `.json` ontology files to `.ttl` ontologies
Use dbpedia ontologies from [text2kgbench](https://zenodo.org/records/7916716) (/data/dbpedia/webnlg/ontologies/)
and convert them to `.ttl` ontologies to use them with this project

Run `json_to_ttl.py`:
```bash
python json_to_ttl.py path/to/json path/to/output
```

---

### 1. Convert `.jsonl` files to `.txt`

Use datasets from [text2kgbench](https://zenodo.org/records/7916716) and convert the `.jsonl` files into plain text files before running the extraction pipeline.

Run `jsonl_to_txt.py`:

```bash
python jsonl_to_txt.py path/to/jsonl path/to/outputdir
```


---

## 2. Run extraction pipelines

After generating the `.txt` files, execute one of the available extraction pipelines.

Run `run.py`:

```bash
python run.py path/to/txts path/to/outputdir --pipeline corenlp
```

---

## Available Pipelines

| Pipeline               | Description                                 |
|------------------------|---------------------------------------------|
| `corenlp`              | Stanford CoreNLP OpenIE extraction          |
| `genie`                | GenIE extraction pipeline                   |
| `corenlp_with_linking` | CoreNLP extraction + DBpedia entity linking |
| `genie_with_linking`   | GenIE extraction + DBpedia entity linking   |

If a pipeline with linking is used, DBPEDIA_ANNOTATE_URL is required in the [.env](./../../../env) file
at the text-pipelines root

---

### 3. Convert ground truth triples to N-Triples

Before evaluating, convert the JSON-lines ground truth file into N-Triples format so it can be used as a reference KG.

Run `groundtruth_to_nt.py`:

```bash
python groundtruth_to_nt.py path/to/triples.jsonl path/to/ontology.ttl path/to/output.nt
```

| Argument        | Description                                      |
|-----------------|--------------------------------------------------|
| `data_file`     | Path to the JSON-lines ground truth triple file  |
| `ontology_file` | Path to the RDF/OWL ontology file                |
| `output_file`   | Output `.nt` file                                |

---

### 4. Evaluate extraction results

After running a pipeline, evaluate the output knowledge graph against a reference KG.

Run `evaluate.py`:

```bash
python evaluate.py \
  --kg-path path/to/kg.nt \
  --reference-kg path/to/reference.nt \
  --output-dir path/to/results/
```

| Argument        | Description                                          |
|-----------------|------------------------------------------------------|
| `--kg-path`     | Path to the extracted knowledge graph file           |
| `--reference-kg`| Path to the reference knowledge graph for comparison |
| `--output-dir`  | Directory where `eval_results.json` will be written  |

The evaluation computes the following metrics and writes them to `eval_results.json`:

| Metric                   | Description                                            |
|--------------------------|--------------------------------------------------------|
| `CountMetric`            | Counts entities and triples in the KG                  |
| `EntityAlignmentMetric`  | Aligns entities to the reference KG via label embedding|
| `DuplicateMetric`        | Detects duplicate entities                             |
| `TripleAlignmentMetric`  | Aligns triples against the reference KG                |

---

## Examples

### CoreNLP

```bash
python run.py ./data/txt ./output --pipeline corenlp
```

### GenIE

```bash
python run.py ./data/txt ./output --pipeline genie
```

### CoreNLP with Entity Linking

```bash
python run.py ./data/txt ./output --pipeline corenlp_with_linking
```

### GenIE with Entity Linking

```bash
python run.py ./data/txt ./output --pipeline genie_with_linking
```

### Ground Truth Conversion

```bash
python groundtruth_to_nt.py ./data/triples.jsonl ./data/ontology.ttl ./data/reference.nt
```

### Evaluation

```bash
python evaluate.py \
  --kg-path ./output/kg.nt \
  --reference-kg ./data/reference.nt \
  --output-dir ./output/eval/
```

---

## Workflow Overview

```text
.jsonl (ground truth)          .jsonl (dataset)
        ↓                              ↓
groundtruth_to_nt.py          jsonl_to_txt.py
        ↓                              ↓
  reference.nt                       .txt
        │                              ↓
        │                           run.py
        │                              ↓
        └──────────► evaluate.py ◄─── kg.nt
                          ↓
                   eval_results.json
```