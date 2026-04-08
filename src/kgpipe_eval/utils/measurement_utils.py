from pydantic import BaseModel

class BinaryClassificationMeasurement(BaseModel):
    tp: int
    fp: int
    tn: int
    fn: int

    def accuracy(self) -> float:
        denom = (self.tp + self.tn + self.fp + self.fn)
        return (self.tp + self.tn) / denom if denom else 0.0
    
    def precision(self) -> float:
        denom = (self.tp + self.fp)
        return self.tp / denom if denom else 0.0
    
    def recall(self) -> float:
        denom = (self.tp + self.fn)
        return self.tp / denom if denom else 0.0
    
    def f1_score(self) -> float:
        p = self.precision()
        r = self.recall()
        denom = (p + r)
        return 2 * p * r / denom if denom else 0.0

    def __str__(self):
        return f"tp: {self.tp}, fp: {self.fp}, tn: {self.tn}, fn: {self.fn}, accuracy: {self.accuracy()}, precision: {self.precision()}, recall: {self.recall()}, f1_score: {self.f1_score()}"

    def to_dict(self) -> dict:
        """
        Convenience export including derived measures.

        Note: do not override BaseModel internals like `__dict__`.
        """
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1_score(),
        }

BCMeasurement = BinaryClassificationMeasurement