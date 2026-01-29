
def test_openie6_task_defintion():
    from text_pipelines.text_tasks import openie6_task_docker
    assert openie6_task_docker.name == "openie6_task_docker"

def test_pipeline_defintion():
    from text_pipelines.text_pipes import openie6_pipe

def test_openie6_pipe():
    from text_pipelines.text_pipes import openie6_pipe
    openie6_pipe()
    