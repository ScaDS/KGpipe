#python run_all_pipes.py --output_dir results

import argparse
import os.path
from pathlib import Path

from dotenv import load_dotenv

from kgpipe.common import Data, DataFormat


def run_pipe(pipe_func, pipe_name: str, input_path: Path, base_output_dir: Path):

    from text_pipelines.text_tasks import te_json_triple_exchange

    pipe_output_dir = base_output_dir / pipe_name
    pipe_output_dir.mkdir(parents=True, exist_ok=True)

    te_json_path = pipe_output_dir / "output.te.json"

    if os.path.isdir(te_json_path):
        te_json_path = next(te_json_path.glob("*.te.json"), None)

    csv_path = pipe_output_dir / "output.csv"

    pipe_func(str(input_path), str(te_json_path))

    data_source = Data(str(te_json_path), DataFormat.TE_JSON)
    data_output = Data(str(csv_path), DataFormat.CSV)

    te_json_triple_exchange.run(
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

    from text_pipelines.text_pipes import (
        genie_pipe,
        graphene_pipe,
        imojie_pipe,
        minie_pipe,
        openie6_pipe,
        genie_pipe_with_linking,
        graphene_pipe_with_linking,
        imojie_pipe_with_linking,
        minie_pipe_with_linking,
        openie6_pipe_with_linking,
    )

    pipes = {
        #"genie": genie_pipe,
        #"graphene": graphene_pipe,
        #"imojie": imojie_pipe,
        #"minie": minie_pipe,
        #"openie6": openie6_pipe,
        "genie_with_linking": genie_pipe,
        "graphene_with_linking": graphene_pipe_with_linking,
        "imojie_with_linking": imojie_pipe_with_linking,
        "minie_with_linking": minie_pipe_with_linking,
        "openie6_with_linking": openie6_pipe_with_linking,
    }

    for name, pipe_func in pipes.items():
        run_pipe(pipe_func, name, input_path, output_dir)


if __name__ == "__main__":
    main()
