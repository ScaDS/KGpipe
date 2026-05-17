import argparse
import os

from kgpipe_tasks.text_processing import corenlp_openie_extraction, corenlp_exchange, label_alias_embedding_rl, \
    dbpedia_spotlight_ner_nel, dbpedia_spotlight_exchange
from kgpipe_tasks.transform_interop import aggregate3_te_json

from text_pipelines import text_pipes
from text_pipelines.text_tasks import genie_task_docker, genie_exchange

def run(input_path, output_path, pipeline):

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
            aggregate3_te_json
        ],

        "genie_with_linking": [
            genie_task_docker,
            genie_exchange,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
            dbpedia_spotlight_exchange,
            aggregate3_te_json
        ]
    }

    os.makedirs(output_path, exist_ok=True)

    if pipeline not in pipelines:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    text_pipes.run_pipe(
        str(input_path),
        str(output_path),
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
        "--pipeline",
        choices=[
            "corenlp",
            "genie",
            "corenlp_with_linking",
            "genie_with_linking"
        ],
        default="corenlp",
        help="Select which pipeline to use"
    )

    args = parser.parse_args()

    run(args.input_dir, args.output_dir, args.pipeline)
