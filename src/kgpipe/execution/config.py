from kgcore.config import KGConfig, ConfigLoader

GLOBAL_STATE = {}

class KgPipeConfig(KGConfig):
    """
    The configuration for kgpipe.
    """
    SYS_KG_URL: str = "memory://"
    SYS_KG_USR: str = ""
    SYS_KG_PSW: str = ""

def load_config():
    """
    Load the configuration for kgpipe.
    """
    config = ConfigLoader("kgpipe").load_config(KgPipeConfig)
    return config