from pathlib import Path

from kgpipe.common import Data, DataFormat, KgTask
from typing import Dict
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document
import json


def _load_er_document(path: Path) -> ER_Document:
    """Parse ER JSON; empty or whitespace-only files yield an empty document (stub tasks may touch-only outputs)."""
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return ER_Document()
    return ER_Document(**json.loads(raw))


def aggregate_matching_results_function(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    er1 = _load_er_document(Path(inputs["json1"].path))
    er2 = _load_er_document(Path(inputs["json2"].path))
    er_comb = ER_Document(matches=er1.matches + er2.matches)
    with open(outputs["output"].path, "w") as f:
        json.dump(er_comb.model_dump(), f, indent=4)


aggregate_matching_results_task = KgTask(
    name="aggregate_matching_results",
    input_spec=dict({"json1": DataFormat.ER_JSON, "json2": DataFormat.ER_JSON}),
    output_spec=dict({"output": DataFormat.ER_JSON}),
    function=aggregate_matching_results_function
)