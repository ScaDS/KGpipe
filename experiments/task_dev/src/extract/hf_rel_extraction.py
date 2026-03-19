from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("zhiheng-huang/bert-base-uncased-tacred")
model = AutoModelForSequenceClassification.from_pretrained("zhiheng-huang/bert-base-uncased-tacred")

text = "[E1]Marie Curie[/E1] won the [E2]Nobel Prize[/E2] in 1903."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# from transformers import pipeline

# re = pipeline("relation-extraction", model="tacred/bert-base-cased")
# text = "Marie Curie won the Nobel Prize in Physics in 1903."

# triples = re(text)
# print(triples)