from kgpipe.common import Registry, DataFormat, Data
from typing import List, Tuple, Dict, Callable
import os
from pathlib import Path
from pydantic import BaseModel
from kgpipe_llm.common import get_config_from_env
from kgpipe_llm.common.core import get_client_from_env
from kgpipe_tasks.transform_interop.exchange.text_extraction import TE_Document, TE_Triple, TE_Span

class Triple(BaseModel):
    head: str
    relation: str
    tail: str

class Triples(BaseModel):
    triples: List[Triple]

# @Registry.task(
#     description="Extract triple pattern from text using a LLM",
#     input_spec={"text": DataFormat.TEXT},
#     output_spec={"triples": DataFormat.TE_JSON},
#     category=["Text", "Triple Extract"]
# )


def apply_to_file_or_files_in_dir(func: Callable, input_path: Path, output_path: Path, parallel: int = 1) -> None:
    from multiprocessing import Pool

    if os.path.isfile(input_path):
        func(input_path, output_path)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        if parallel > 1:
            file_pairs = [
                (os.path.join(input_path, file), os.path.join(output_path, file.replace(".txt", ".json")))
                for file in os.listdir(input_path)
            ]

            with Pool(parallel) as p:
                p.starmap(func, file_pairs)
            p.join()
            p.close()
        else:
            for file in os.listdir(input_path):
                func(os.path.join(input_path, file), os.path.join(output_path, file))

class TripleExtract_v1():

    @staticmethod
    def extract_triples(text: str) -> Triples:
        prompt = """
Extract triple patterns in the form [ {{ "head": "head", "relation": "relation", "tail": "tail" }} ] from the following text:

{text}
""".format(text=text)

        system_prompt = "You are a KG engineer. Generating JSON-LD output for a specific task."
        llm = get_client_from_env()

        response = llm.send_prompt(prompt, Triples, system_prompt)

        return Triples(**response)
 
    @staticmethod
    def extract_te_document(text: str) -> TE_Document:
        triples_simples = TripleExtract_v1.extract_triples(text)

        def get_start_end_of_substring(text: str, substring: str) -> Tuple[int, int]:
            start = text.find(substring)
            end = start + len(substring)
            return start, end

        triples = []
        for triple in triples_simples.triples:
            subject_start, subject_end = get_start_end_of_substring(text, triple.head)
            predicate_start, predicate_end = get_start_end_of_substring(text, triple.relation)
            object_start, object_end = get_start_end_of_substring(text, triple.tail)

            subject_span = TE_Span(start=subject_start, end=subject_end, surface_form=triple.head)
            predicate_span = TE_Span(start=predicate_start, end=predicate_end, surface_form=triple.relation)
            object_span = TE_Span(start=object_start, end=object_end, surface_form=triple.tail)

            triples.append(TE_Triple(subject=subject_span, predicate=predicate_span, object=object_span))
        
        return TE_Document(text=text, triples=triples)

    @staticmethod
    def extract_txt_file_to_te_json_file(input_path: str, output_path: Path) -> None:
        text = open(input_path, "r").read()
        te_document = TripleExtract_v1.extract_te_document(text)
        with open(output_path, "w") as f:
            f.write(te_document.model_dump_json(indent=2))

@Registry.task(
    description="Extract triple pattern from text using a LLM",
    input_spec={"input": DataFormat.TEXT},
    output_spec={"output": DataFormat.TE_JSON},
    category=["Text", "Triple Extract"]
)
def llm_task_text_triple_extract_v1(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    apply_to_file_or_files_in_dir(TripleExtract_v1.extract_txt_file_to_te_json_file, input_path, output_path, 8)