from pathlib import Path
from typing import Dict, List, Optional
import json
import re

from kgpipe.common import KG
from kgpipe.common.models import Data, KgPipePlan, DataFormat
from kgpipe.evaluation.aspects import reference
from kgpipe.evaluation.aspects.reference import Reference


def resolve_relative_path(path: str, base_path: Path) -> Path:
   return (base_path / path).resolve()

def get_source_name(source: Data) -> str:
   return source.path.parent.name if source.path.is_dir() or source.path.name.startswith("data") else source.path.stem


# TODO extract entity type from by using list metadata_path to get all ending with metadata_name
def get_expected_source_entities(plan: KgPipePlan, testing_path: Path, metadata_name: str = "entities.txt") -> Optional[Reference]:

    source = plan.source
    source_name = get_source_name(source)

    print(source_name)

    # find the same source name in the testing dir tree
    metadata_dir = testing_path / source_name
    # find the source entities file
    source_entities_file = metadata_dir / metadata_name
    # read the source entities file
    if source_entities_file.exists():
        return Reference(data=Data(path=source_entities_file, format=DataFormat.TEXT), trust_level=1, associated_data={})
    else:
        print(f"No source entities file found for {source_entities_file}")
    return None


class ReferenceManager():
   def __init__(self, plan: KgPipePlan, testing_path: Path, reference_names: List[str]):
      self.plan = plan
      self.testing_path = testing_path
      self.reference_names = reference_names
      self.references = self.get_references()

   def load_reference(self, name: str) -> Optional[Reference]:
      for reference in self.references:
         if reference.data.path.name == name:
            return reference
      return None

   def get_references(self) -> List[Reference]:
      references = []
      source_entities = get_expected_source_entities(self.plan, self.testing_path)
      if source_entities:
         references.append(source_entities)
      return references

   def get_by_name(self, name: str) -> Optional[Reference]:
      for reference in self.references:
         if reference.data.path.name == name:
            return reference
      return None

class StageResultManager():

   def __init__(self, stage_path: Path):
      self.stage_path = stage_path
      self.plan = self.get_plan()
      self.kg = self.get_kg()

   def get_kg(self) -> KG:
      if (self.stage_path / "result.nt").exists():
         kg = KG(id=self.stage_path.name, name=self.stage_path.name, path=self.stage_path / "result.nt", format=DataFormat.RDF_NTRIPLES)
      elif (self.stage_path / "result.ttl").exists():
         kg = KG(id=self.stage_path.name, name=self.stage_path.name, path=self.stage_path / "result.ttl", format=DataFormat.RDF_TTL)
      else:
         raise ValueError(f"No result KG file found in {self.stage_path}")
      kg.plan = self.plan
      return kg

   # TODO is bad change exec-plan.json to contain the source
   # TODO use pydantic model, possibly is already defined somewhere else
#    def get_references(self) -> Dict:
#       def get_expected_source_entities(plan: KgPipePlan) -> List[str]:
#          entities = set()
#          source_regex = re.compile(r".*/((\d_)?source\.\w+).*")
#          source_names = []
#          for step in plan.steps:
#             for data in step.input:
#                match = source_regex.match(str(data.path))
#                if match:
#                   source_names.append(match.group(1).replace(".", "-"))
#          testing_path = self.stage_path.parent / "../testing/entities"
#          for source_name in source_names:
#             entities_path = (testing_path / f"{source_name}_film-entities.txt").resolve()
#             if entities_path.exists():
#                with open(entities_path, "r") as f:
#                   for line in f:
#                      entities.add(line.strip())
#          return list(entities)
         

#       return {
#          "expected_source_entities": get_expected_source_entities(self.plan)
#       }


   def get_plan(self) -> KgPipePlan: # TODO: use KgTask
      with open(self.stage_path / "exec-plan.json", "r") as f:
         json_data =  json.load(f)
         return KgPipePlan(**json_data)


