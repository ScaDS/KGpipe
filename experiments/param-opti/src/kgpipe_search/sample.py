
class TripleSampleIterator:

    def __init__(self):

    def __iter__(self):
        return self

    def __next__(self) -> List[Tuple[str, str, str]]:
        if self.budget <= 0:
            raise StopIteration
        self.budget -= 1
        return self.next()

    def next(self) -> List[Tuple[str, str, str]]:
        return random.sample(self.config_space, self.budget)

