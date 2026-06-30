import sys
import os
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    if len(sys.argv) < 3:
        print("Usage: rebel_cli.sh <input> <output>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if os.path.exists(input_path):
        run(input_path, output_path)
    else:
        print("Input path does not exist")


def extract_triples(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
    return triplets


def run(input_path, output_path):
    model_name = "Babelscape/rebel-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            run(
                os.path.join(input_path, file),
                os.path.join(output_path, file),
            )

    elif os.path.isfile(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            input_text = f.read()

        print(input_text)
        inputs = tokenizer(input_text, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        print(text)

        triples = extract_triples(text)

        print(triples)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(triples))


if __name__ == "__main__":
    main()