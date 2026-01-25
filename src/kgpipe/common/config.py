from kgcore.config import KGConfig, ConfigLoader

GLOBAL_STATE = {}

class KgPipeConfig(KGConfig):
    """
    The configuration for kgpipe.
    """
    SYS_KG_URL: str = "memory://"
    SYS_KG_USR: str = ""
    SYS_KG_PSW: str = ""


SOURCE_NAMESPACE: str = "http://kg.org/rdf/"
TARGET_RESOURCE_NAMESPACE: str = "http://kg.org/resource/"
TARGET_ONTOLOGY_NAMESPACE: str = "http://kg.org/ontology/"

def load_config() -> KgPipeConfig:
    """
    Load the configuration for kgpipe.
    """
    config = ConfigLoader("kgpipe").load_config(KgPipeConfig)
    return config

config = load_config()