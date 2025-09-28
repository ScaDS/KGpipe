from kgpipe_llm.common.api_utils import get_token_count

# from typing import Type, Optional, Dict, Any
# from pydantic import BaseModel


# import os
# from dotenv import load_dotenv

# load_dotenv()


# from typing import List, Optional
# from pydantic import BaseModel, Field
# from enum import Enum

# class Category(str, Enum):
#     fiction = "fiction"
#     nonfiction = "nonfiction"

# class Book(BaseModel):
#     """Details about a book."""
#     title: str = Field(..., description="Book title")
#     author: str = Field(..., description="Primary author")
#     year: Optional[int] = Field(None, ge=0, description="Publication year")
#     categories: List[Category] = Field(default_factory=list)

# api_key = os.getenv("YOUR_OPENAI_API_KEY")
# if api_key is None:
#     raise ValueError("OPENAI_API_KEY is not set")

# # response = requests.post(
# #     "https://api.openai.com/v1/chat/completions",
# #     headers={
# #         "Content-Type": "application/json",
# #         "Authorization": f"Bearer {api_key}"
# #     },
# #     json={
# #         "model": "gpt-5-nano",
# #         "messages": [
# #             {"role": "system", "content": "You are a helpful assistant."},
# #             {"role": "user", "content": "What is the capital of France?"}
# #         ]
# #     }
# # )

# # print(json.dumps(response.json(), indent=4))


# # raw_args, book = call_with_tool(
# #     endpoint_url="https://api.openai.com/v1/chat/completions",
# #     api_key=api_key,
# #     model_name="gpt-5-nano",
# #     user_content="Give me structured details for 'The Left Hand of Darkness' by Ursula K. Le Guin.",
# #     pyd_model=Book,
# # )

# # print("Raw tool args:", raw_args)           # dict from the model
# # print("Validated pydantic object:", book)   # Book(title=..., author=..., ...)

def test_token_count():
    text = "Hello, world!"
    print(get_token_count(text))

from rdflib import Graph

def test_calc_input_token_for_kg_construction():
    #kg_path = "/home/marvin/project/data/old/current/workdir/data.ttl"
    # kg_path = "/home/marvin/project/data/old/current/workdir/data_lt.nt"
    # kg_data = open(kg_path, "r").read()

    # graph = Graph()    
    # graph.bind("dbr", "http://dbpedia.org/resource/")
    # graph.bind("dbp", "http://dbpedia.org/property/")
    # graph.bind("dbo", "http://dbpedia.org/ontology/")
    # graph.bind("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    # graph.bind("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
    # graph.bind("xsd", "http://www.w3.org/2001/XMLSchema#")
    # graph.bind("owl", "http://www.w3.org/2002/07/owl#")
    # graph.parse(kg_path, format="turtle")

    # kg_data = graph.serialize(format="turtle")
    # graph.serialize(destination=kg_path.replace(".nt", ".ttl"), format="turtle")

    string = "Hello, world!"
    string2 = "Hello,\nworld!\n\n"

    print(get_token_count(string))
    print(get_token_count(string2))

if __name__ == "__main__":
    test_token_count()