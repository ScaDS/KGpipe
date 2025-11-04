import psycopg2
from kgback.back import EntityCentricKnowledgeGraphBackend
from kgcore.model.graph import Triple, Node, Literal



class PostgresECB(EntityCentricKnowledgeGraphBackend):
    """
    Postgres backend for the knowledge graph.
    triple table: subject: long, predicate: long, object: long
    id table: id: long, name: varchar(255)
    """
    
    def __init__(self, host="localhost", port=5432, database="kglab", user="postgres", password="mysecretpassword"):
        self.conn = psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # if not exists
        self.__execute(f"CREATE TABLE IF NOT EXISTS triples (subject BIGINT, predicate BIGINT, object BIGINT)")
        # unique on uri
        self.__execute(f"CREATE TABLE IF NOT EXISTS entities (id BIGINT PRIMARY KEY, uri VARCHAR(255) UNIQUE)")
        # auto increment on id
        self.__execute(f"CREATE SEQUENCE IF NOT EXISTS entity_id_seq START 1")
        self.__execute(f"ALTER TABLE entities ALTER COLUMN id SET DEFAULT nextval('entity_id_seq')")

    def __close(self):
        self.cursor.close()
        self.conn.close()

    def __execute(self, query):
        self.cursor.execute(query)


    # def get_entity_triples(self, entity_uri):
    #     """
    #     For a given entity uri, return all triples that contain the entity.
    #     """
    #     self.__execute(f"SELECT subject, predicate, object FROM triples WHERE subject = (SELECT id FROM entities WHERE uri = '{entity_uri}') OR object = (SELECT id FROM entities WHERE uri = '{entity_uri}')")
    #     return self.cursor.fetchall()


    def get_or_create_entity_id(self, uri: str) -> int:
        """
        Insert the entity if it doesn't exist, or return its existing id.
        """
        query = f"""
            INSERT INTO entities (uri)
            VALUES ('{uri}')
            ON CONFLICT (uri) DO UPDATE SET uri = EXCLUDED.uri
            RETURNING id
        """.format(uri)
        self.__execute(query)
        entity_id = self.cursor.fetchone()[0]
        return entity_id
        
    def add_triples(self, triples: list[Triple]):
        """
        For a given entity uri, add all triples that contain the entity.
        This will crate the id mappings for all URIs in the triples.
        """
        uris = []
        for triple in triples:
            uris.append(triple.subject.uri)
            uris.append(triple.predicate.uri)
            uris.append(triple.object.uri)
        uris = list(set(uris))
        uri_to_id = {}
        for uri in uris:
            uri_to_id[uri] = self.get_or_create_entity_id(uri)

        for triple in triples:
            self.__execute(f"INSERT INTO triples (subject, predicate, object) VALUES ({uri_to_id[triple.subject.uri]}, {uri_to_id[triple.predicate.uri]}, {uri_to_id[triple.object.uri]})")
        self.conn.commit()



    def get_all_triples(self):
        """
        Return all triples in the database with IDs replaced by URIs.
        """
        query = """
            SELECT es.uri AS subject_uri,
                ep.uri AS predicate_uri,
                eo.uri AS object_uri
            FROM triples t
            JOIN entities es ON t.subject = es.id
            JOIN entities ep ON t.predicate = ep.id
            JOIN entities eo ON t.object = eo.id
            ORDER BY es.uri, ep.uri, eo.uri
        """
        # assuming __execute returns rows (list of tuples or list of dicts)
        self.cursor.execute(query)

        triples = []
        for row in self.cursor.fetchall():
            triples.append(Triple(subject=Node(uri=row[0]), predicate=Node(uri=row[1]), object=Node(uri=row[2])))
        return triples

    def get_triples_by_subject(self, subject_uri: str):
        """
        Return all triples with the given subject uri.
        """
        query = f"""
            SELECT es.uri AS subject_uri, ep.uri AS predicate_uri, eo.uri AS object_uri FROM triples t JOIN entities es ON t.subject = es.id JOIN entities ep ON t.predicate = ep.id JOIN entities eo ON t.object = eo.id WHERE es.uri = '{subject_uri}' ORDER BY es.uri, ep.uri, eo.uri
        """
        self.cursor.execute(query)
        triples = []
        for row in self.cursor.fetchall():
            triples.append(Triple(subject=Node(uri=row[0]), predicate=Node(uri=row[1]), object=Node(uri=row[2])))
        return triples

    def clear_database(self):
        """
        Clear the database.
        """
        self.__execute(f"DELETE FROM triples")
        self.__execute(f"DELETE FROM entities")
        self.__execute(f"ALTER SEQUENCE entity_id_seq RESTART WITH 1")
        self.conn.commit()