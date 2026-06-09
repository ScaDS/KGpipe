from kgpipe.common import DataFormat, Registry

from kgpipe_llm.tasks import llm_task_text_triple_extract_v1
from kgpipe_llm.common.core import get_client_from_env
from pydantic import BaseModel
from typing import List
from typing import Dict
from kgpipe.common import Data



# Extract triple patterns in the form [ {{ "head": "head", "relation": "relation", "tail": "tail" }} ] from the following text:
# {text}
@Registry.task(input_spec={"input": DataFormat.TEXT}, output_spec={"output": DataFormat.TE_JSON})
def text_extraction_direct(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    IN_FORMAT=DataFormat.TEXT
    text = "John Smith is a software engineer at Google."
    prompt = """
Extract triple patterns in the form [ {{ "head": "head", "relation": "relation", "tail": "tail" }} ] from the following text:

{text}
""".format(text=text)

    system_prompt = "You are a KG engineer. Generating JSON-LD output for a specific task."
    llm = get_client_from_env()

    response = llm.send_prompt(prompt, Triples, system_prompt)

    print(response)


# @Registry.task(input_spec={"input": DataFormat.TEXT}, output_spec={"output": DataFormat.TE_JSON})
# def test_text_extraction_target():
    
#     prompt = """
# Extract triple patterns in the form [ {{ "head": "head", "relation": "relation", "tail": "tail" }} ] from the following text:

# {text}
# """.format(text=text)
#     pass

@Registry.task(input_spec={"input": DataFormat.TEXT}, output_spec={"output": DataFormat.TE_JSON})
def test_text_construction_direct():
    

    pass

@Registry.task(input_spec={"input": DataFormat.TEXT}, output_spec={"output": DataFormat.TE_JSON})
def test_text_construction_target():    
    IN_FORMAT=DataFormat.TEXT
    pass

# @Registry.task(input_spec={"input": DataFormat.TEXT}, output_spec={"output": DataFormat.TE_JSON})
# def test_entity_linking_embedd():
#     pass

# @Registry.task(input_spec={"input": DataFormat.TEXT}, output_spec={"output": DataFormat.TE_JSON})
# def test_relation_linking_embedd():
#     ONTOLOGY=""
#     pass

# @Registry.task(input_spec={"input": DataFormat.TEXT}, output_spec={"output": DataFormat.TE_JSON})
# def test_type_extraction():
#     pass
