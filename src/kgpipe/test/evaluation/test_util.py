
import pytest

from kgpipe.evaluation.util import StageResultManager, ReferenceManager
from kgpipe.test.util import get_test_data_path

@pytest.mark.skip(reason="Not implemented")
def test_stage_result_file_manager():
   stage_path = get_test_data_path("inc/pipeline_1/stage_1")
   srm = StageResultManager(stage_path)
   assert srm

@pytest.mark.skip(reason="Not implemented")
def test_reference_manager():
   pass

@pytest.mark.skip(reason="Not implemented")
def test_get_expected_source_entities():
   pass