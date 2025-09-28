# from geneval import CONTEXT
# import yaml
# from pathlib import Path
# import json
# from geneval.pipelines import load_pipeline_catalog

# from . import get_test_data_path

# def test_context():
#     assert CONTEXT.data_dir == "TODO"
#     assert CONTEXT.result_dir == "TODO"
#     assert CONTEXT.temp_dir == "TODO"

# def test_pipeline_conf():
#     pipeline_conf = load_pipeline_catalog(get_test_data_path("pipeline.conf"))
#     print(pipeline_conf.model_dump())   

#     assert pipeline_conf.root["dummy"].tasks == ["dummy_task"]
#     assert pipeline_conf.root["dummy_reuse"].tasks == ["dummy"]

#     assert pipeline_conf.root["dummy"].description == "Dummy pipeline"
#     assert pipeline_conf.root["dummy_reuse"].description == "Dummy pipeline with reuse of dummy"


#     # # assert pipeline_conf.pipelines["dummy"].description == "Dummy pipeline"
#     # assert pipeline_conf["dummy"].tasks == ["dummy_task"]
#     # # assert pipeline_conf.pipelines["dummy_reuse"].description == "Dummy pipeline with reuse of dummy"
#     # assert pipeline_conf["dummy_reuse"].tasks == ["dummy"]