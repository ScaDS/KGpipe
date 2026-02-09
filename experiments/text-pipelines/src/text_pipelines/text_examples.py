
def test_openie6_task_defintion():
    from text_pipelines.text_tasks import openie6_task_docker
    assert openie6_task_docker.name == "openie6_task_docker"

def test_pipeline_defintion():
    from text_pipelines.text_pipes import openie6_pipe

def test_openie6_pipe():
    from text_pipelines.text_pipes import openie6_pipe
    openie6_pipe()

def test_graphene_output_to_json_te():
    from text_pipelines.text_tasks import graphene_nt_exchange
    pass

def test_graphene_pipe():
    from text_pipelines.text_pipes import graphene_pipe
    graphene_pipe("input.txt", "output.json")


def test_minie_pipe():
    from text_pipelines.text_pipes import minie_pipe
    minie_pipe("/home/theo/Work/SCADS.AI/Projects/KGpipe/experiments/text-pipelines/wrappers/minie/input.txt",
               "/home/theo/Work/SCADS.AI/Projects/KGpipe/experiments/text-pipelines/wrappers/minie/output.json")