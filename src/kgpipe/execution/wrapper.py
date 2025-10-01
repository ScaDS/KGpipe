from kgpipe.common.models import KgTask, Data
from typing import List

class ExecutionWrapper:
    def __init__(self, task: KgTask):
        self.task = task
    
    def execute(self, inputs: List[Data], outputs: List[Data]) -> None:
        pass

class LocalExecutionWrapper(ExecutionWrapper):

    def execute(self, inputs: List[Data], outputs: List[Data]) -> None:
        pass

class HttpExecutionWrapper(ExecutionWrapper):
    
    endpoint: str
    
    def __init__(self, task: KgTask, endpoint: str):
        super().__init__(task)
        self.endpoint = endpoint
    
    def execute(self, inputs: List[Data], outputs: List[Data]) -> None:
        pass

class DockerExecutionWrapper(ExecutionWrapper):
    def execute(self, inputs: List[Data], outputs: List[Data]) -> None:
        pass

