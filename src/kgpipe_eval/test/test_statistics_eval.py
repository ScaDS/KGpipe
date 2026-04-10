from kgpipe_eval.metrics.statistics import CountMetric
from kgpipe_eval.test.utils import get_test_kg
from kgpipe_eval.utils.kg_utils import KgManager

def test_count_metric():
    metric = CountMetric()
    report = metric.compute(KgManager.load_kg(get_test_kg()))
    print(report)