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


