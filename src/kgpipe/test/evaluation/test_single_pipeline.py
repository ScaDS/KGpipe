# import pytest
# from kgpipe.common import KG, DataFormat
# from pathlib import Path

# from kgpipe.evaluation import Evaluator, EvaluationConfig, EvaluationAspect
# from kgpipe.evaluation.writer import render_as_table, render_as_table_multi

# ROOT = Path(__file__).resolve().parents[3]
# TD = ROOT / "geneval" / "test_data"

# kg0 = KG("0", name="seed", path=TD / "kg0.ttl", format=DataFormat.RDF_TTL)
# kg1a = KG("1a", name="kg1a", path=TD / "kg1a.ttl", format=DataFormat.RDF_TTL)
# kg1b = KG("1b", name="kg1b", path=TD / "kg1b.ttl", format=DataFormat.RDF_TTL)

# @pytest.mark.skip(reason="Verbose")
# def test_ssp_evaluation():
#     config = EvaluationConfig([EvaluationAspect.STATISTICAL, EvaluationAspect.SEMANTIC, EvaluationAspect.REFERENCE])

#     evaluator = Evaluator(config)
    
#     report = evaluator.evaluate(kg0)
#     render_as_table(report, show_details=False)


# @pytest.mark.skip(reason="Verbose")
# def test_multi_ssp_report():
#     config = EvaluationConfig([EvaluationAspect.STATISTICAL, EvaluationAspect.SEMANTIC, EvaluationAspect.REFERENCE])

#     kgs = [kg0, kg1a, kg1b]

#     evaluator = Evaluator(config)
#     reports = []
#     for kg in kgs:
#         report = evaluator.evaluate(kg)
#         reports.append(report)
    
#     render_as_table_multi(reports, show_details=False, cell_render_func=lambda m: f"{m.value:.3f}")

    