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

---

## Workflow Overview

```text
.jsonl
   ↓
jsonl_to_txt.py
   ↓
.txt
   ↓
run.py
   ↓
Triples / Linked Entities / JSON Output
```
