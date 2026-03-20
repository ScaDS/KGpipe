#python run_all_pipes.py --output_dir results

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from kgpipe.common import Data, DataFormat

from text_pipelines import text_pipes

def run_pipe(pipe_tasks, pipe_name: str, input_path: Path, base_output_dir: Path):

    from text_pipelines.text_tasks import linked_te_json_triple_exchange

    pipe_output_dir = base_output_dir / pipe_name
    pipe_output_dir.mkdir(parents=True, exist_ok=True)

    te_json_path = pipe_output_dir / "output.te.json"
    csv_path = pipe_output_dir / "output.csv"

    text_pipes.run_pipe(str(input_path), str(te_json_path), pipe_tasks)

    while te_json_path.is_dir():
        te_json_path = next(te_json_path.glob("*.te.json"), None)

    data_source = Data(str(te_json_path), DataFormat.TE_JSON)
    data_output = Data(str(csv_path), DataFormat.CSV)

    linked_te_json_triple_exchange.run(
        [data_source],
        [data_output],
        stable_files_override=True
    )

    print(f"{pipe_name} done! Output: {pipe_output_dir}")


def main():
    text_pipelines_folder_path = Path(__file__).parent.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="test")
    parser.add_argument(
        "--input",
        default=str(text_pipelines_folder_path / "test/Titanic.txt")
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(dotenv_path=text_pipelines_folder_path / ".env")

    from text_pipelines.text_tasks import imojie_task_docker, imojie_exchange, minie_task_docker, minie_exchange, \
        graphene_task_docker, graphene_nt_exchange, genie_task_docker, genie_exchange
    from kgpipe_tasks.text_processing import corenlp_openie_extraction, label_alias_embedding_rl, \
        dbpedia_spotlight_ner_nel, dbpedia_spotlight_exchange, corenlp_exchange
    from kgpipe_tasks.transform_interop import aggregate3_te_json

    pipes = {
        #"genie": genie_pipe,
        #"graphene": graphene_pipe,
        #"imojie": imojie_pipe,
        #"minie": minie_pipe,
        #"openie6": openie6_pipe,
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
            ],
        "graphene_with_linking": [
            graphene_task_docker,
            graphene_nt_exchange,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
             dbpedia_spotlight_exchange,
            aggregate3_te_json],
        "imojie_with_linking": [
            imojie_task_docker,
            imojie_exchange,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
            dbpedia_spotlight_exchange,
            aggregate3_te_json
        ],
        "minie_with_linking": [
            minie_task_docker,
            minie_exchange,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
            dbpedia_spotlight_exchange,
            aggregate3_te_json
        ],
        #"openie6_with_linking": [
        #    openie6_task_docker,
        #    label_alias_embedding_rl,
        #    dbpedia_spotlight_ner_nel,
        #    dbpedia_spotlight_exchange,
        #    aggregate3_te_json
        #],
    }

    for name, pipe_tasks in pipes.items():
        run_pipe(pipe_tasks, name, input_path, output_dir)

def run_dataset_pipe(pipe_tasks, pipe_name: str, input_path: Path, base_output_dir: Path):

    from text_pipelines.text_tasks import linked_te_json_triple_exchange

    pipe_output_dir = base_output_dir / pipe_name
    pipe_output_dir.mkdir(parents=True, exist_ok=True)

    text_pipes.run_pipe(str(input_path), str(pipe_output_dir), pipe_tasks)

    print(f"{pipe_name} done! Output: {pipe_output_dir}")

def run_dataset():

    text_pipelines_folder_path = Path(__file__).parent.parent.parent

    load_dotenv(dotenv_path=text_pipelines_folder_path / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="test")

    args = parser.parse_args()
    input_path = Path(os.getenv("DATASET_PATH")) / "split_0" / "sources" / "text" / "data"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from text_pipelines.text_tasks import imojie_task_docker, imojie_exchange, minie_task_docker, minie_exchange, \
        graphene_task_docker, graphene_nt_exchange, genie_task_docker, genie_exchange
    from kgpipe_tasks.text_processing import label_alias_embedding_rl, dbpedia_spotlight_ner_nel, \
        dbpedia_spotlight_exchange, corenlp_exchange
    from kgpipe_tasks.transform_interop import aggregate3_te_json

    pipes = {
        #"genie": genie_pipe,
        #"graphene": graphene_pipe,
        #"imojie": imojie_pipe,
        #"minie": minie_pipe,
        #"openie6": openie6_pipe,
        #"corenlp_with_linking": [
        #    corenlp_openie_extraction,
        #    corenlp_exchange,
        #    label_alias_embedding_rl,
        #    dbpedia_spotlight_ner_nel,
        #    dbpedia_spotlight_exchange,
        #    aggregate3_te_json
        #    ],
        "genie_with_linking": [
            genie_task_docker,
            genie_exchange,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
            dbpedia_spotlight_exchange,
            aggregate3_te_json
            ],
        "graphene_with_linking": [
            graphene_task_docker,
            graphene_nt_exchange,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
             dbpedia_spotlight_exchange,
            aggregate3_te_json],
        "imojie_with_linking": [
            imojie_task_docker,
            imojie_exchange,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
            dbpedia_spotlight_exchange,
            aggregate3_te_json
        ],
        "minie_with_linking": [
            minie_task_docker,
            minie_exchange,
            label_alias_embedding_rl,
            dbpedia_spotlight_ner_nel,
            dbpedia_spotlight_exchange,
            aggregate3_te_json
        ],
        #"openie6_with_linking": [
        #    openie6_task_docker,
        #    label_alias_embedding_rl,
        #    dbpedia_spotlight_ner_nel,
        #    dbpedia_spotlight_exchange,
        #    aggregate3_te_json
        #],
    }

    for name, pipe_tasks in pipes.items():
        run_dataset_pipe(pipe_tasks, name, input_path, output_dir)

if __name__ == "__main__":
    #main()
    run_dataset()
