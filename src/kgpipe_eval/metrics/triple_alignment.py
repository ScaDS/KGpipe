from pydantic import BaseModel, ConfigDict
from typing import Literal

from kgpipe.common import KG
from kgpipe_eval.metrics.entity_alignment import EntityAlignmentConfig
from kgpipe_eval.utils.kg_utils import KgLike, KgManager, TripleGraph
from kgpipe_eval.utils.alignment_utils import align_triples_by_value_embedding
from kgpipe_eval.utils.measurement_utils import BCMeasurement
from kgpipe_eval.api import Measurement, Metric, MetricResult

# measures precision, recall, f1 score, etc.

class TripleAlignmentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    reference_kg: KgLike
    method: Literal["value_embedding", "exact"] = "value_embedding"
    entity_alignment_config: EntityAlignmentConfig
    value_sim_threshold: float = 0.5
    cache_literal_embeddings: bool = False
    cache_ref_literal_embeddings: bool = True

def eval_triple_alignment(tg: TripleGraph, config: TripleAlignmentConfig):
    if config.method == "value_embedding":
        alignments = align_triples_by_value_embedding(tg, config)
    elif config.method == "exact":
        pass
        # alignments = align_triples_by_exact_match(tg, config)
    else:
        raise ValueError(f"Invalid method: {config.method}")

    print("Triple alignments: ", len(alignments))

    ref_tg = KgManager.load_kg(config.reference_kg)
    ref_triples = set(ref_tg.triples((None, None, None)))
    gen_triples = set(tg.triples((None, None, None)))

    aligned_ref_triples = set(a.target for a in alignments)
    aligned_gen_triples = set(a.source for a in alignments)

    tp = len(aligned_ref_triples)  # aligned reference triples
    fp = len(gen_triples - aligned_gen_triples)  # generated triples not aligned to any reference triple
    tn = 0
    fn = len(ref_triples - aligned_ref_triples)  # reference triples missing in generation

    return BCMeasurement(tp=tp, fp=fp, tn=tn, fn=fn)

# def eval_triple_alignment_by_label_embedding(method: Literal["exact", "fuzzy", "semantic"] = "exact"):
#     pass


# def eval_triple_alignment_by_label_embedding_soft_literals(method: Literal["exact", "fuzzy", "semantic"] = "exact"):
#     pass

class ReferenceTripleAlignmentMetric(Metric):
    
    def compute(self, kg: KG, config: TripleAlignmentConfig):
        m: BCMeasurement = eval_triple_alignment(kg, config)
        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="tp", value=m.tp, unit="number"),
                Measurement(name="fp", value=m.fp, unit="number"),
                Measurement(name="tn", value=m.tn, unit="number"),
                Measurement(name="fn", value=m.fn, unit="number"),
                Measurement(name="precision", value=m.precision(), unit="percentage"),
                Measurement(name="recall", value=m.recall(), unit="percentage"),
                Measurement(name="f1_score", value=m.f1_score(), unit="percentage"),
            ],
            summary=f"Triple alignment by {config.method}",
        )


# Backward-compatibility alias (imported by `kgpipe_eval.metrics.__init__`).
TripleAlignmentMetric = ReferenceTripleAlignmentMetric