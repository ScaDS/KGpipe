# README.md
## Build Docker
```bash
docker build -t genie .
```

## Run Docker
```bash
docker run --rm \
  -v sentences.txt:/data/input.txt \
  -v output.json:/data/output.json \
  genie genie.sh /data/input.txt /data/output.json
```


