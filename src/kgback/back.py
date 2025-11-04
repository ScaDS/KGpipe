from kgcore.model.graph import Triple, Node, Literal
from kgback.config import KgBackConfig

class KnowledgeGraphBackend():
    def get_triples(self):
        pass

    def add_triples(self, triples):
        pass


class KnowledgeGraphBackend():
    pass

class EntityCentricKnowledgeGraphBackend(KnowledgeGraphBackend):
    def get_entity_triples(self, node: Node):
        pass

    def add_triples(self, triples: list[Triple]):
        pass

    def get_all_triples(self):
        pass

def get_backend(config: KgBackConfig):
    if config.db_url.startswith("postgres"):
        from kgback.postgres_back import PostgresECB
        return PostgresECB(config.db_url)
    elif config.db_url.startswith("sqlite"):
        from kgback.sqlite_back import SqliteECB
        return SqliteECB(config.db_url)
    else:
        raise ValueError(f"Invalid database type: {config.db_type}")