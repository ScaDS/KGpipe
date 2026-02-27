from kgcore.config import KGConfig, ConfigLoader

GLOBAL_STATE = {}

class KgPipeConfig(KGConfig):
    """
    The configuration for kgpipe.
    """
    SYS_KG_URL: str = "sparql://localhost:8890/sparql-auth" #"memory://"
    SYS_KG_USR: str = "dba"
    SYS_KG_PSW: str = "mysecret"
    ONTOLOGY_PREFIX: str = "http://github.com/ScaDS/kgpipe/ontology/"
    PIPEKG_PREFIX: str = "http://github.com/ScaDS/kgpipe/resource/"


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