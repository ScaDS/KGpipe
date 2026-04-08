from kgpipe.common import KG

from kgpipe_eval.api import Metric, Measurement, MetricResult

from kgpipe_eval.utils.measurement_utils import BCMeasurement
from kgpipe_eval.utils.alignment_utils import align_entities_by_label_embedding, EntityAlignmentConfig, load_entity_uri_label_type_pairs, get_entity_uri_label_type_pairs

# Core Interface

def eval_entity_alignment(kg: KG, config: EntityAlignmentConfig):
    if config.method == "label_embedding":
        alignments = eval_entity_alignment_by_label_embedding(kg, config)
    elif config.method == "label_alias_embedding":
        alignments = eval_entity_alignment_by_label_alias_embedding(kg, config)
    elif config.method == "label_embedding_and_type":
        alignments = eval_entity_alignment_by_label_embedding_and_type(kg, config)
    else:
        raise ValueError(f"Invalid method: {config.method}")
    return alignments

# Specific Implementations

def eval_entity_alignment_by_label_embedding_and_type(kg: KG, config: EntityAlignmentConfig):
    alignments = align_entities_by_label_embedding(kg, config)

    ref_entity_uri_label_type_pairs = load_entity_uri_label_type_pairs(config)
    gen_entity_uri_label_type_pairs = list(get_entity_uri_label_type_pairs(kg))

    ref_types = {pair.uri: pair.type for pair in ref_entity_uri_label_type_pairs if pair.type is not None}
    # TODO gen_types can be multiple types, we need to handle this
    gen_types = {pair.uri: pair.type for pair in gen_entity_uri_label_type_pairs if pair.type is not None}

    filtered_alignments = []
    for alignment in alignments:
        if alignment.target in ref_types and alignment.source in gen_types:
            if ref_types[alignment.target] == gen_types[alignment.source]:
                filtered_alignments.append(alignment)

    ref_uris = set(pair.uri for pair in ref_entity_uri_label_type_pairs)
    gen_uris = set(pair.uri for pair in gen_entity_uri_label_type_pairs)
    aligned_gen_uris = set(alignment.target for alignment in filtered_alignments)
    aligned_ref_uris = set(alignment.source for alignment in filtered_alignments)

    tp = len(ref_uris & aligned_gen_uris) # generated entities that are also in the reference
    fp = len(gen_uris - aligned_ref_uris) # generated entities that are not in the reference
    tn = 0
    fn = len(ref_uris - aligned_gen_uris) # missing generated entities that are in the reference

    return BCMeasurement(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn
    )

def eval_entity_alignment_by_label_embedding(kg: KG, config: EntityAlignmentConfig):
    alignments = align_entities_by_label_embedding(kg, config)

    ref_entity_uri_label_type_pairs = load_entity_uri_label_type_pairs(config)
    gen_entity_uri_label_type_pairs = list(get_entity_uri_label_type_pairs(kg))

    ref_uris = set(pair.uri for pair in ref_entity_uri_label_type_pairs)
    gen_uris = set(pair.uri for pair in gen_entity_uri_label_type_pairs)
    aligned_gen_uris = set(alignment.target for alignment in alignments)
    aligned_ref_uris = set(alignment.source for alignment in alignments)

    tp = len(ref_uris & aligned_gen_uris) # generated entities that are also in the reference
    fp = len(gen_uris - aligned_ref_uris) # generated entities that are not in the reference
    tn = 0
    fn = len(ref_uris - aligned_gen_uris) # missing generated entities that are in the reference

    return BCMeasurement(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn
    )

def eval_entity_alignment_by_label_alias_embedding(kg: KG, config: EntityAlignmentConfig):
    raise NotImplementedError("Label alias embedding alignment is not implemented yet")

# Metric Implementation

class EntityAlignmentMetric(Metric):
    def compute(self, kg: KG, config: EntityAlignmentConfig):
        alignments: BCMeasurement = eval_entity_alignment(kg, config)
        return MetricResult(
            metric=self,
            measurements=[
                Measurement(name="tp", value=alignments.tp, unit="number"),
                Measurement(name="fp", value=alignments.fp, unit="number"),
                Measurement(name="tn", value=alignments.tn, unit="number"),
                Measurement(name="fn", value=alignments.fn, unit="number"),
                Measurement(name="precision", value=alignments.precision(), unit="percentage"),
                Measurement(name="recall", value=alignments.recall(), unit="percentage"),
                Measurement(name="f1_score", value=alignments.f1_score(), unit="percentage"),
            ],
            summary=f"Entity alignment by {config.method}"
        )