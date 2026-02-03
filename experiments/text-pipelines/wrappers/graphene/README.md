# README.md
## Build Docker
```bash
docker build -t graphene .
```

## Run Docker
```bash
docker run --rm \
  -v sentences.txt:/data/input.txt \
  -v predictions.txt:/data/output.txt \
  graphene graphene.sh /data/input.txt /data/output.txt
```


