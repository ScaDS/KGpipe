# README.md
## Build Docker
```bash
docker build -t graphene .
```

## Run Docker
```bash
docker run --rm \
  -v sentences.txt:/data/input.txt \
  -v output.nt:/data/output.nt \
  graphene graphene.sh /data/input.txt /data/output.nt
```

## Tool Parameters

### CLI Parameters

#### Operation (Choose one)
  - `COREF` – Coreference Resolution
  - `SIM` – Discourse Simplification
  - `RE` – Relation Extraction

---

#### Input (Choose one)
  - `TEXT`
  - `FILE`

---

#### Output (Choose one)
  - `CMDLINE`
  - `FILE`

---

### Output Format Parameters

#### Coreference Format (Choose one)
  - `DEFAULT`
  - `SERIALIZED`

#### Discourse Simplification Format (Choose one)
  - `DEFAULT`
  - `DEFAULT_RESOLVED`
  - `FLAT`
  - `FLAT_RESOLVED`
  - `SERIALIZED`

#### Relation Extraction Format (Choose one)
  - `DEFAULT`
  - `DEFAULT_RESOLVED`
  - `FLAT`
  - `FLAT_RESOLVED`
  - `RDF`
  - `SERIALIZED`

---

### Processing Flags
- `doCoreference`
- `isolateSentences`

---

### Coreference Configuration (Choose one)
- `StanfordCoref`
- `PyCobaltCoref`

---

### Relation Extraction Configuration

#### Extractor (Choose one)
- `HeadRelationExtractor`
- `NestedRelationExtractor`

#### Boolean Parameters
- `exploit-core`
- `exploit-contexts`
- `separate-noun-based`
- `separate-purposes`
- `separate-attributions`