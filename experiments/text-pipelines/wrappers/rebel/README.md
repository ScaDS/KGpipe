# README.md
## Build Docker
```bash
docker build -t rebel .
```

## Run Docker
```bash
sudo docker run --rm \
  -v /home/rebel/input:/data/input \
  -v /home/rebel/output:/data/output \
  rebel rebel.sh /data/input /data/output
```


