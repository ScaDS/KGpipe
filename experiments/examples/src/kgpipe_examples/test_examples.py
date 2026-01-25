

def test_python_task_defintion():
    from kgpipe_examples.task_examples import pipe_task_python
    assert pipe_task_python.name     == "pipe_task_python"

def test_docker_task_defintion():
    from kgpipe_examples.task_examples import pipe_task_docker
    assert pipe_task_docker.name == "pipe_task_docker"

def test_remote_task_defintion():
    from kgpipe_examples.task_examples import pipe_task_remote
    assert pipe_task_remote.name == "pipe_task_remote"

def test_pipeline_defintion():
    from kgpipe_examples.pipe_examples import pipe_example
    