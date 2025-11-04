
docker_build:
	docker build -t kgpipe .

docker_help:
	docker run --rm kgpipe --help