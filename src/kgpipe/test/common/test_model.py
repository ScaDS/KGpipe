from kgpipe.common.models import KgPipePlan, KgPipePlanStep, Data, DataFormat
from pathlib import Path
import json

def test_kg_pipe_plan():
   plan = KgPipePlan(
      steps=[
         KgPipePlanStep(
            task="paris_entity_matching",
            input=[Data(path=Path("data.nt"), format=DataFormat.RDF_NTRIPLES)],
            output=[Data(path=Path("data.paris_csv"), format=DataFormat.PARIS_CSV)]
         ),
         KgPipePlanStep(
            task="paris_csv_to_matching_format",
            input=[Data(path=Path("data.paris_csv"), format=DataFormat.PARIS_CSV)],
            output=[Data(path=Path("data.em_json"), format=DataFormat.ER_JSON)]
         ),
      ],
      seed=Data(path=Path("seed.nt"), format=DataFormat.RDF_NTRIPLES),
      source=Data(path=Path("source.nt"), format=DataFormat.RDF_NTRIPLES),
      result=Data(path=Path("result.nt"), format=DataFormat.RDF_NTRIPLES),
   )
   
   plan_json = plan.model_dump_json()
   print(plan_json)

   plan_back = KgPipePlan(**json.loads(plan_json))

   assert plan == plan_back