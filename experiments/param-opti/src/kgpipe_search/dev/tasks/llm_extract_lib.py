from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_llm_extract():
    # Load the Flan-T5 Large checkpoint (780M parameters)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    prompt = """Extract the organizations and locations from the following text:
    "Sarah flew from Berlin to Leipzig to attend a workshop at the university." """

    # Encode the prompt and generate extraction
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)

    # Decode the output
    extracted_info = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(extracted_info)

