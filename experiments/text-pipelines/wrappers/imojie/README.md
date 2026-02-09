# README.md
## Build Docker
```bash
docker build -t imojie .
```

## Run Docker
```bash
docker run --rm \
  -v sentences.txt:/data/input.txt \
  -v output.txt:/data/output.txt \
  imojie imojie.sh /data/input.txt /data/output.txt
```


