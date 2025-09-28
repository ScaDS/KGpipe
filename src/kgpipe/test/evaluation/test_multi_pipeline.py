# from kgpipe.common.models import KgPipePlan, DataFormat 
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# import json
# import re
# from enum import Enum
# import pytest
# from kgpipe.test.util import get_test_data_path
# from kgpipe.common import KG

# from kgpipe.evaluation.aspects.reference import Reference, ER_EntityMatchMetric, TE_EntityLinkingMetric, TE_RefName
# from kgpipe.evaluation.util import ReferenceManager, StageResultManager


# def last_em_json_path(plan: KgPipePlan, base_path: Path) -> Optional[Path]:
#    for step in reversed(plan.steps):
#       for output in step.output:
#          if output.format == DataFormat.ER_JSON:
#             return (base_path / output.path).resolve()
#    return None

# def last_ie_json_path(plan: KgPipePlan, base_path: Path) -> Optional[Path]:
#    for step in reversed(plan.steps):
#       for output in step.output:
#          if output.format == DataFormat.TE_JSON:
#             return (base_path / output.path).resolve()
#    return None

# class RefName(Enum):
#    ENTITIES = "entities.txt"
#    FILM_ENTITIES = "film_entities.txt"

# reference_names = [RefName.ENTITIES.value, RefName.FILM_ENTITIES.value]


# def em_evaluation(kg: KG, em_json_path: Path, rm: ReferenceManager):
   
#    from kgpipe.evaluation.aspects.reference import ER_RefName

#    em_ref = rm.get_by_name(RefName.ENTITIES.value)

#    if em_json_path.exists() and em_ref:
#       print("Start entity matching evaluation")
#       em_metric = ER_EntityMatchMetric()

#       ref_dict = {
#          ER_RefName.MATCHES.value: em_ref,
#          ER_RefName.GT_MATCH_CLUSTERS.value: em_ref,
#       }
      
#       em_metric.compute(kg, ref_dict)

#    else:
#       print(f"No entity matching evaluation possible: em_json = {em_json_path.exists()}"+
#       f" entities.txt = {rm.get_by_name(RefName.ENTITIES.value)}")

# def el_evaluation(kg: KG, el_json_path: Path, rm: ReferenceManager):

#    em_ref = rm.get_by_name(RefName.ENTITIES.value)

#    if el_json_path.exists() and em_ref:
#       print("Start entity matching evaluation")

#       el_metric = TE_EntityLinkingMetric()

#       ref_dict = {
#          TE_RefName.MATCHES.value: em_ref,
#          TE_RefName.GT_MATCH_CLUSTERS.value: em_ref,
#       }

#       el_metric.compute(kg, ref_dict)
   
#    else:
#       print(f"No entity matching evaluation possible: el_json = {el_json_path.exists()}"+
#       f" entities.txt = {rm.get_by_name(RefName.ENTITIES.value)}")

# @pytest.mark.skip(reason="Not implemented")
# def test_em_el_evaluation():

#    stage_path = get_test_data_path("inc/pipeline_1/stage_1")
#    testing_path = get_test_data_path("inc/testing")
#    srm = StageResultManager(stage_path)

#    plan = srm.get_plan()   

#    rm = ReferenceManager(plan, testing_path=testing_path, reference_names = reference_names)

#    em_json_path = last_em_json_path(plan, stage_path)
#    ie_json_path = last_ie_json_path(plan, stage_path)

#    if em_json_path and ie_json_path:
#       raise ValueError("Undecided:Both em_json and ie_json found")
#    elif em_json_path:
#       em_evaluation(srm.kg, em_json_path, rm)
#    elif ie_json_path:
#       el_evaluation(srm.kg, ie_json_path, rm)
#    else:
#       raise ValueError("No em_json or ie_json found")
