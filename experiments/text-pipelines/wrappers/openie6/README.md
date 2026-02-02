# README.md
## Build Docker
```bash
docker build -t openie6 .
```

## Run Docker
```bash
docker run --rm --gpus all \
  -v /local-data/sentences.txt:/data/input.txt \
  -v /local-data/predictions.txt:/data/output.txt \
  openie6.sh /data/input.txt /data/output.txt
```


