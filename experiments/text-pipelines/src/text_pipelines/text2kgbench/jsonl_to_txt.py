import json
import os
import argparse


def jsonl_to_txt(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

                file_id = data["id"]
                text = data["sent"]

                output_path = os.path.join(output_dir, f"{file_id}.txt")

                with open(output_path, "w", encoding="utf-8") as out_file:
                    out_file.write(text)

                print(f"Created: {output_path}")

            except Exception as e:
                print(f"Error in line {line_number}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts JSONL to TXT-Files."
    )

    parser.add_argument(
        "input_file",
        help="Path to JSONL"
    )

    parser.add_argument(
        "output_dir",
        help="Path to output directory"
    )

    args = parser.parse_args()

    jsonl_to_txt(args.input_file, args.output_dir)