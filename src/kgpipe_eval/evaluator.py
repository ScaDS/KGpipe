from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import traceback

from kgpipe_eval.api import Metric, MetricResult
from kgpipe_eval.utils.kg_utils import TripleGraph


def _metric_key(metric: Metric) -> str:
    return getattr(metric, "key", metric.__class__.__name__)


@dataclass
class Evaluator:
    """
    Execute multiple metrics against a KG and pass the right config (if any).
    """

    def run(
        self,
        kg: TripleGraph,
        metrics: Sequence[Metric],
        confs: Mapping[str, Any] | None = None,
    ) -> List[MetricResult]:
        confs = dict(confs or {})
        results: List[MetricResult] = []

        for metric in metrics:
            key = _metric_key(metric)
            cfg = confs.get(key, confs.get(key.lower()))

            compute = getattr(metric, "compute", None)
            if compute is None:
                raise TypeError(f"Metric {key!r} has no compute() method")

            sig = inspect.signature(compute)
            # Bound method: typically (kg) or (kg, config)
            params = [
                p for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]

            try:
                if len(params) <= 1:
                    # compute(self) or compute(self, kg) -- call without config
                    res = compute(kg) if len(params) == 1 else compute()
                else:
                    # compute(self, kg, config, ...)
                    if cfg is None:
                        raise KeyError(
                            f"Missing config for metric {key!r}. "
                            f"Provide `confs[{key!r}]`."
                        )
                    res = compute(kg, cfg)
            except Exception as e:
                print(f"Failed running metric {key!r}: {e}")
                print(traceback.format_exc())
                raise RuntimeError(f"Failed running metric {key!r}") from e

            if not isinstance(res, MetricResult):
                raise TypeError(f"Metric {key!r} returned {type(res)!r}, expected MetricResult")
            results.append(res)

        return results

