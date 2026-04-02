

# @dataclass(frozen=True)
# class BinaryClassificationStats:
#     tp: int
#     fp: int
#     tn: int
#     fn: int

#     def recall(self) -> float:
#         d = self.tp + self.fn
#         return self.tp / d if d else 0.0

#     def precision(self) -> float:
#         d = self.tp + self.fp
#         return self.tp / d if d else 0.0

#     def f1(self) -> float:
#         p = self.precision()
#         r = self.recall()
#         return 2 * p * r / (p + r) if (p + r) else 0.0

#     def accuracy(self) -> float:
#         d = self.tp + self.fp + self.tn + self.fn
#         return (self.tp + self.tn) / d if d else 0.0

# def reference_binary_classification(kg: KgKg, config: MetricConfig) -> MetricResult:
#     stats = BinaryClassificationStats(tp=10, fp=5, tn=15, fn=3)
#     return MetricResult(
#         metric_key="reference_binary_classification",
#         summary="Reference comparison computed",
#         measurements=[
#             Measurement("tp", stats.tp),
#             Measurement("fp", stats.fp),
#             Measurement("tn", stats.tn),
#             Measurement("fn", stats.fn),
#             Measurement("precision", stats.precision(), "ratio"),
#             Measurement("recall", stats.recall(), "ratio"),
#             Measurement("f1", stats.f1(), "ratio"),
#             Measurement("accuracy", stats.accuracy(), "ratio"),
#         ],
#     )

# def graph_size(kg: KgKg, config: MetricConfig) -> MetricResult:
#     size = 1532
#     return MetricResult(
#         metric_key="graph_size",
#         measurements=[
#             Measurement("triple_count", size, "triples")
#         ],
#         summary=f"Graph contains {size} triples",
#     )

# def entity_duplication_rate(kg: KgKg, config: MetricConfig) -> MetricResult:
#     duplicates = 7
#     total = 100
#     rate = duplicates / total if total else 0.0
#     return MetricResult(
#         metric_key="entity_duplication_rate",
#         measurements=[
#             Measurement("duplication_rate", rate, "ratio"),
#             Measurement("duplicate_entities", duplicates, "entities"),
#             Measurement("total_entities", total, "entities"),
#         ],
#         summary=f"Entity duplication rate: {rate:.2%}",
#     )
# ---

# class BinaryClassifier():
#     tp: int
#     fp: int
#     tn: int
#     fn: int

#     def recall(self) -> float:
#         return self.tp / (self.tp + self.fn)
    
#     def precision(self) -> float:
#         return self.tp / (self.tp + self.fp)
    
#     def f1(self) -> float:
#         return 2 * self.precision() * self.recall() / (self.precision() + self.recall())
    
#     def accuracy(self) -> float:
#         return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

# @lru_cache
# def compute_binary_classifier(kg: KgKg) -> BinaryClassifier:
#     return BinaryClassifier(tp=10, fp=5, tn=15, fn=3)


# # Option 1 the metrics are recall, precision, f1, accuracy
# def reference_recall(kg: KgKg, reference: KgKg) -> KgMetricResult:
#     binary_classifier = compute_binary_classifier(kg, reference)
#     return KgMetricResult(summary=f"Reference recall: {binary_classifier.recall()}")

# def reference_precision(kg: KgKg, reference: KgKg) -> KgMetricResult:
#     binary_classifier = compute_binary_classifier(kg, reference)
#     return KgMetricResult(summary=f"Reference precision: {binary_classifier.precision()}")

# def reference_f1(kg: KgKg, reference: KgKg) -> KgMetricResult:
#     binary_classifier = compute_binary_classifier(kg, reference)
#     return KgMetricResult(summary=f"Reference F1: {binary_classifier.f1()}")

# #Option 2 the metrics are Binary Classification which allows for more detailed analysis
# def reference_binary_classification(kg: KgKg, reference: KgKg) -> KgMetricResult:
#     binary_classifier = compute_binary_classifier(kg, reference)
#     return KgMetricResult(summary=f"Reference binary classification: {binary_classifier.tp}, {binary_classifier.fp}, {binary_classifier.tn}, {binary_classifier.fn}")