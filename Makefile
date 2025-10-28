
build_docker:
	docker build -t kgpipe --build-context kgcore=../kgcore/ --build-context kgback=../kgback/ --build-context pyodibel=../odibel .

build_docker_nc:
	docker build -t kgpipe 	--no-cache --build-context kgcore=../kgcore/ --build-context pyodibel=../odibel .



# time docker run --rm kgpipe bash -c 'cd experiments/moviekg; pytest -s -v src/moviekg/pipelines'
