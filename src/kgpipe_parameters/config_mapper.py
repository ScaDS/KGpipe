
"""
Maps a GLOBAL configuration to a local Parameter of a task implementation.
"""

from kgpipe.common.model.configuration import Parameter, ConfigurationProfile
from kgpipe.common.model.task import KgTask, Data, TaskInput, TaskOutput, KgTask

class ConfigMapper:
    def __init__(self, task: KgTask):
        self.task = task

    def map_config(self, config: ConfigurationMapping):
        return self.task.config




def example_task(i: TaskInput, o: TaskOutput, p: ConfigurationProfile):
    pass