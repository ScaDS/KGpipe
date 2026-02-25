# README.md
## Build Docker
```bash
docker build -t minie .
```

## Run Docker
```bash
docker run --rm \
  -v input.txt:/data/input.txt \
  -v output.txt:/data/output.txt \
  minie minie.sh /data/input.txt /data/output.txt
```


## Tool Parameters

### Extraction Mode
- `MinIE.Mode.SAFE` – conservative, high-precision extractions
- `MinIE.Mode.DICTIONARY` – dictionary-based extraction (requires multi-word dictionaries)
- `MinIE.Mode.AGGRESSIVE` – aggressive extraction, may produce more triples
- `MinIE.Mode.COMPLETE` – extract all possible candidates

### Multi-word Dictionary (for DICTIONARY mode)
- `dictionaryFiles` – paths to multi-word collocation files