from pydantic import BaseModel

class BinaryClassificationMeasurement(BaseModel):
    tp: int
    fp: int
    tn: int
    fn: int

    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)
    
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn)
    
    def f1_score(self) -> float:
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def __str__(self):
        return f"tp: {self.tp}, fp: {self.fp}, tn: {self.tn}, fn: {self.fn}, accuracy: {self.accuracy()}, precision: {self.precision()}, recall: {self.recall()}, f1_score: {self.f1_score()}"

    def __dict__(self):
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1_score()
        }

BCMeasurement = BinaryClassificationMeasurement