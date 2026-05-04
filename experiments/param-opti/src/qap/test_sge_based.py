def eval_paris_pipeline(entity_matching_threshold: float, relation_matching_threshold: float):
    """
    evaluate a paris pipeline with different thresholds for entity matching and relation matching
    """
    pass

    # ref_kg_path = Path("data/inputs/reference_kg/data_agg.nt")
    # gen_kg_path = Path(f"data/tmp/rdf_pipelines/result_{entity_matching_threshold}_{relation_matching_threshold}.nt")

    # source_grounded_correctness_config = SourceGroundedCorrectnessConfig(
    #     kg_graph=ref_kg_path,
    #     source_corpus=gen_kg_path,
    #     index_dir=Path("data/tmp/source_grounded_correctness"),
    #     verbalize_method="natural",
    #     verifier="nli",
    #     nli_model="facebook/bart-large-mnli",
    #     nli_device="cpu",
    #     llm_model="gpt-4.1-mini",
    #     llm_device="cpu"
    # )

    # source_grounded_correctness_metric = SourceGroundedCorrectnessMetric()
    # source_grounded_correctness_metric.compute(KgManager.load_kg(gen_kg_path), source_grounded_correctness_config)

def eval_openie_pipeline():
    """
    evaluate an openie pipeline
    """
    pass

    # ref_kg_path = Path("data/inputs/reference_kg/data_agg.nt")
    # gen_kg_path = Path(f"data/tmp/rdf_pipelines/result_{entity_matching_threshold}_{relation_matching_threshold}.nt")

    # source_grounded_coverage_config = SourceGroundedCoverageConfig(
    #     kg_graph=ref_kg_path,
    #     source_corpus=gen_kg_path,