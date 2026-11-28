"""Sample pipeline defined with the public Python API.

The task list and parameter values match ``experiments/examples/sample.yaml``.
``Pipeline.plan()`` and ``Pipeline.from_yaml`` compile to the same fingerprint.
"""

from pathlib import Path
from typing import Optional, Union

from kgpipe import Pipeline

PathLike = Union[str, Path]


def sample_pipeline(
    source: PathLike,
    seed: PathLike,
    result: PathLike,
    data_dir: PathLike,
    *,
    base_dir: Optional[Path] = None,
) -> Pipeline:
    """The ``sample`` catalog entry, written in Python."""
    return Pipeline(
        name="sample",
        description="Align source RDF with a seed KG",
        tasks=[
            {
                "id": "match",
                "task": "dummy_match",
                "backend": "python",
                "parameters": {"entity_matching_threshold": 0.8},
            },
            "dummy_exchange",
            "dummy_fuse",
            "dummy_type",
        ],
        parameters={"entity_matching_threshold": 0.99},
        connect="auto",
        backend="python",
        inputs={"source": "nt", "seed": "nt"},
        output={"result": "nt"},
        source=source,
        seed=seed,
        result=result,
        data_dir=data_dir,
        base_dir=base_dir,
    )
