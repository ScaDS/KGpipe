import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from kgpipe.common import DataFormat
from kgpipe_tasks.construction.construct import construct_rdf_from_te_json_mappings_only
from kgpipe_tasks.text_processing import corenlp_openie_extraction, corenlp_exchange, label_alias_embedding_rl, \
    dbpedia_spotlight_ner_nel, dbpedia_spotlight_exchange
from kgpipe_tasks.transform_interop import aggregate3_te_json

from text_pipelines import text_pipes
from text_pipelines.text_tasks import genie_task_docker, genie_exchange, rebel_task_docker

def run(input_path, output_path, seed_path, pipeline):
    pipelines = {
        "corenlp": [
            corenlp_openie_extraction,
            corenlp_exchange
        ],

        "genie": [
            genie_task_docker,
            genie_exchange
        ],

        "corenlp_with_linking": [
            corenlp_openie_extraction,
            corenlp_exchange,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
            dbpedia_spotlight_exchange,
            aggregate3_te_json,
            construct_rdf_from_te_json_mappings_only
        ],

        "genie_with_linking": [
            genie_task_docker,
            genie_exchange,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
            dbpedia_spotlight_exchange,
            aggregate3_te_json,
            construct_rdf_from_te_json_mappings_only
        ],

        "rebel_with_linking": [
            rebel_task_docker,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
            dbpedia_spotlight_exchange,
            aggregate3_te_json,
            construct_rdf_from_te_json_mappings_only
        ]
    }

    os.makedirs(output_path, exist_ok=True)

    if pipeline not in pipelines:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    if "linking" in pipeline:
        text_pipelines_folder_path = Path(__file__).parent.parent.parent.parent
        load_dotenv(dotenv_path=text_pipelines_folder_path / ".env")

        os.environ["ONTOLOGY_PATH"] = seed_path
        text_pipes.run_pipe(
            str(input_path),
            str(output_path),
            seed_path,
            DataFormat.TEXT,
            DataFormat.RDF_NTRIPLES,
            pipelines[pipeline]
        )
    else:
        os.environ["ONTOLOGY_PATH"] = seed_path
        text_pipes.run_pipe(
            str(input_path),
            str(output_path),
            seed_path,
            DataFormat.TEXT,
            DataFormat.TE_JSON,
            pipelines[pipeline]
        )
    print(f"Done using '{pipeline}'! Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs extraction pipelines on TXT files."
    )

    parser.add_argument(
        "input_dir",
        help="Path to txt folder"
    )

    parser.add_argument(
        "output_dir",
        help="Path to output directory"
    )

    parser.add_argument(
        "seed",
        help="Path to seed file"
    )

    parser.add_argument(
        "--pipeline",
        choices=[
            "corenlp",
            "genie",
            "corenlp_with_linking",
            "genie_with_linking",
            "rebel_with_linking"
        ],
        default="corenlp",
        help="Select which pipeline to use"
    )

    args = parser.parse_args()

    run(args.input_dir, args.output_dir, args.seed, args.pipeline)
