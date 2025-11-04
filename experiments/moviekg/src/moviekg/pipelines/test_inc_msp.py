
import pytest
from itertools import permutations
from _pytest.compat import NotSetType
from typing import Sequence

from moviekg.pipelines.helpers import run_helper
from moviekg.config import ssp, OUTPUT_ROOT

def idfn(param):
    # param is a tuple like ("text", "json", "rdf")
    if isinstance(param, tuple):
        return "-".join(param)
    return str(param)


@pytest.mark.parametrize(
    "source_1, source_2, source_3", 
    permutations(list[str](ssp.keys()), 3),
    ids=idfn
)
def test_inc_msp_with_3_splits(source_1, source_2, source_3):
        
    print("-" * 100)
    print(f"Running {source_1}, {source_2}, {source_3}")
    print("-" * 100)

    output_dir = OUTPUT_ROOT / f"{source_1}_{source_2}_{source_3}"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_helper(ssp[source_1], source_1, 1, 1, output_dir)
    run_helper(ssp[source_2], source_2, 2, 2, output_dir)
    run_helper(ssp[source_3], source_3, 3, 3, output_dir)
