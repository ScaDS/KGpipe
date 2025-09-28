import pytest

from kgpipe.evaluation.aspects.reference import ReferenceConfig
from kgpipe.evaluation.util import ReferenceManager, StageResultManager
from kgpipe.evaluation.writer import render_metric_as_table
from kgpipe.test.util import get_test_data_path, get_data, get_test_kg



# TODO there are different kinds of reference metric categories
# - KG and Reference KG
# - Task and References
# - KG and (Expected) Source Reference Information

SHOW_DETAILS = True

config = ReferenceConfig(
    name="test reference config",
    GT_MATCHES=get_test_data_path("inc/testing/gt_matches.csv"),
    ER_MATCH_THRESHOLD=0.5,
    EXPECTED_TEXT_LINKS=get_test_data_path("inc/testing/verified_text_links.json"),
    TE_LINK_THRESHOLD=0.5,
    VERIFIED_SOURCE_ENTITIES=get_test_data_path("inc/testing/verified_source_entities.csv")
)


# =============================================================================
# Test Integration
# =============================================================================

# rm = ReferenceManager(plan=sm.plan, testing_path=get_test_data_path("inc/testing/"))


@pytest.mark.skip(reason="Not implemented yet")
def test_reference_kg_entity_alignment():
    """
    uses paris on both and then replaces the ids of target with the ids of reference
    then checks if the entities are the same
    TODO: case tp, fp, tn, fn for entities and relations
    """
    from kgpipe.evaluation.aspects.reference import ER_EntityMatchMetric
    metric = ER_EntityMatchMetric()
    report = metric.compute(get_test_kg("inc/reference_1.nt"), config=config)
    render_metric_as_table(report, show_details=SHOW_DETAILS)
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_reference_kg_entity_alignment_with_paris():
    """
    uses paris on both and then replaces the ids of target with the ids of reference
    then checks if the entities are the same
    TODO: case tp, fp, tn, fn for entities and relations
    """
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_reference_kg_entity_alignment_with_text_embedding():
    # EntityOverlapMetric
    # TODO: check llm4rml code
    pass

# @pytest.mark.skip(reason="Not implemented yet")
# def test_reference_kg_entity_alignment_with_llm():
#     """
#     LLM
#     Prompt
#     """
#     pass

# @pytest.mark.skip(reason="Not implemented yet")
# def test_reference_triple_alignment():
#     """
#     TODO: move to kgpipe.evaluation.aspects.reference
#     TODO: case tp, fp, tn, fn for entities and relations
#     """
#     # 1 load reference kg
#     reference_kg = get_test_kg("inc/reference_1.nt")
#     # 2 load target kg
#     target_kg = get_test_kg("inc/rdf_0.nt")
#     # 3 align triples (exact match)
#     reference_set = get_triples(reference_kg.get_graph())
#     target_set = get_triples(target_kg.get_graph())
#     # 4 evaluate
#     print("reference set", len(reference_set))
#     print("target set", len(target_set))
#     print("intersection", len(reference_set & target_set))

@pytest.mark.skip(reason="Not implemented yet")
def test_reference_triple_alignment_with_paris():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_reference_triple_alignment_with_text_embedding():
    # TODO: check llm4rml code
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_entity_type_derivation():
    pass

# =============================================================================
# Task Entity/Relation Matches
# =============================================================================

stage_with_er_task_path = get_test_data_path("inc/pipeline_1/stage_1/")
sm1 = StageResultManager(stage_with_er_task_path)
kg_with_er_task = sm1.kg

def test_entity_match_task_quality():
    """
    Tests the quality of the entity match results based on a provided reference ground truth.
    """

    from kgpipe.evaluation.aspects.reference import ER_EntityMatchMetric
    metric = ER_EntityMatchMetric()
    report = metric.compute(kg_with_er_task, config=config)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

def test_relation_match_task_quality():
    """
    Tests the quality of the relation match results based on a provided reference ground truth.
    """
    from kgpipe.evaluation.aspects.reference import ER_RelationMatchMetric
    metric = ER_RelationMatchMetric()
    report = metric.compute(kg_with_er_task, config=config)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

# =============================================================================
# Task Entity and Relation Linking
# =============================================================================

stage_with_te_task_path = get_test_data_path("inc/pipeline_2/stage_1/")
sm2 = StageResultManager(stage_with_te_task_path)
kg_with_te_task = sm2.kg

def test_entity_link_task_quality():
    from kgpipe.evaluation.aspects.reference import TE_ExpectedEntityLinkMetric
    metric = TE_ExpectedEntityLinkMetric()
    report = metric.compute(kg_with_te_task, config=config)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

@pytest.mark.skip(reason="Not implemented")
def test_relation_link_task_quality():
    from kgpipe.evaluation.aspects.reference import TE_ExpectedRelationLinkMetric
    metric = TE_ExpectedRelationLinkMetric()
    report = metric.compute(kg_with_te_task, config=config)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

# =============================================================================
# Source Integration (depends on source metadata, requires traceability)
# =============================================================================

def test_source_entity_coverage():
    from kgpipe.evaluation.aspects.reference import SourceEntityCoverageMetric
    metric = SourceEntityCoverageMetric()
    report = metric.compute(kg_with_er_task, config=config)
    render_metric_as_table(report, show_details=SHOW_DETAILS)

