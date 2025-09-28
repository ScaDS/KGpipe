from pydantic import BaseModel
from typing import List, Optional
import json

class MatchingData(BaseModel):
    """
    A class representing matching data.
    """
    pass

class ExtractionData(BaseModel):
    """
    A class representing extraction data.
    """
    pass

#######################################

# class ER_Entity:
#     id: AnyStr = None
#     type: AnyStr = None

class ER_Match(BaseModel):
    """
    A match between two entities
    """
    id_1: str 
    id_2: str
    score: float
    id_type: Optional[str] = None 

class ER_Block(BaseModel):
    """
    A blocking group of entities
    """
    entities: List[str] = []
    block_key: Optional[str] = None


class ER_Cluster(BaseModel):
    """
    A cluster of entities
    """
    entities: List[str] = []

class ER_Document(BaseModel):
    matches: List[ER_Match] = []
    blocks: List[ER_Block] = []
    clusters: List[ER_Cluster] = []

    def add_blocks(self, entites: List[str], key: str):
        self.blocks.append(ER_Block(entities=entites, block_key=key))

    def add_match(self, match: ER_Match):
        self.matches.append(match)

    def add_cluster(self, cluster: ER_Cluster):
        self.clusters.append(cluster)


# if __name__ == "__main__":
    
#     newTeDoc = aggregate_te_documents([
#         TE_Document(**json.load(open("/data/abstracts_Person.100.openie.tejson/Dennis_Peacock.nt.json"))),
#         TE_Document(**json.load(open("/data/abstracts_Person.100.spotlight.tejson/Dennis_Peacock.nt"))),
#         TE_Document(**json.load(open("/data/abstracts_Person.100.customrl.tejson/Dennis_Peacock.nt.json")))
#     ])

#     print(newTeDoc.model_dump_json(indent=4))

