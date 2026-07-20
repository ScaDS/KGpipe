from abc import ABC, abstractmethod
from kgpipe.common.model.pipeline import KgPipe

class KGpipeExecution(ABC):
    """Base class for KGpipe execution."""

    def __init__(self, pipeline: KGpipePipeline):
        self.pipeline = pipeline

    @abstractmethod
    def execute(self):
        """Execute the pipeline."""
        pass