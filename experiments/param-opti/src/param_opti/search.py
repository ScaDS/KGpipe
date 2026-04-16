

def sample_random_valid(task_impls: List[str]):
    pass

class SearchSpace:
    def __init__(self, task_impls: List[str]):
        self.task_impls = task_impls

class NeighborhoodSearch:
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    def search(self, budget: int):
        pass


