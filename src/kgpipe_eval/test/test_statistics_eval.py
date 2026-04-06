from kgpipe_eval.metrics.statistics import CountMetric
from kgpipe_eval.test.examples import TEST_TURTLE_TRIPLES, REFERENCE_TURTLE_TRIPLES

def test_count_metric():
    metric = CountMetric()
    report = metric.compute(TEST_TURTLE_TRIPLES)
    render_metric_as_table(report, show_details=SHOW_DETAILS)