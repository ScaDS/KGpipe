# README.md
## Build Docker
```bash
docker build -t flant5 .
```

## Run Docker
```bash
sudo docker run --rm \
  -v /home/theodebian-laptop/PycharmProjects/KGpipe/experiments/text-pipelines/wrappers/flant5/test.txt:/data/input.txt \
  -v /home/test/output.json:/data/output.json \
  flant5 flant5.sh /data/input.txt /data/output.json
```


