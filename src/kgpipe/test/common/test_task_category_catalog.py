from kgpipe.common.models import TaskCategoryCatalog


def test_entity_resolution_children_include_expected_subtasks():
    children = TaskCategoryCatalog.get_children("EntityResolution")
    assert "Blocking" in children
    assert "Matching" in children
    assert "EntityMatching" in children
    assert "Clustering" in children


def test_subtask_relationships_for_entity_resolution():
    assert TaskCategoryCatalog.is_subtask_of("Blocking", "EntityResolution")
    assert TaskCategoryCatalog.is_subtask_of("Matching", "EntityResolution")
    assert TaskCategoryCatalog.is_subtask_of("Clustering", "EntityResolution")
    assert not TaskCategoryCatalog.is_subtask_of("EntityResolution", "Blocking")


def test_ancestors_and_descendants_are_resolved():
    ancestors = TaskCategoryCatalog.get_ancestors("EntityMatching")
    descendants = TaskCategoryCatalog.get_descendants("EntityResolution")

    assert ancestors[0] == "EntityResolution"
    assert "TaskCategory" in ancestors
    assert "Blocking" in descendants
    assert "Clustering" in descendants


def test_register_custom_category_under_existing_parent():
    TaskCategoryCatalog.register("CandidateGeneration", parent="EntityResolution")
    assert TaskCategoryCatalog.has("CandidateGeneration")
    assert TaskCategoryCatalog.get_parent("CandidateGeneration") == "EntityResolution"
    assert TaskCategoryCatalog.is_subtask_of("CandidateGeneration", "EntityResolution")
