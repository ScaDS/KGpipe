import sys
import os
import json

from genie.models import GeniePL


DATA_DIR = os.path.join(os.getcwd(), "data")


def load_model():
    ckpt_name = "genie_r.ckpt"
    path_to_checkpoint = os.path.join(DATA_DIR, "models", ckpt_name)

    model = GeniePL.load_from_checkpoint(
        checkpoint_path=path_to_checkpoint
    )

    return model


def main():
    if len(sys.argv) < 3:
        print("Usage: genie.sh <input.txt> <output.txt>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    model = load_model()

    with open(input_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    generation_args = {
        "num_beams": 10,
        "num_return_sequences": 1,
        "return_dict_in_generate": True,
        "output_scores": True,
        "seed": 123
    }

    outputs = model.sample(sentences, **generation_args)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Extraction finished. Results written to {output_path}")


if __name__ == "__main__":
    main()
