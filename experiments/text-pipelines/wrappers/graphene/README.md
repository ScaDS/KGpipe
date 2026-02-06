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


