import sqlite3
from kgback.back import EntityCentricKnowledgeGraphBackend
from kgcore.model.graph import Triple, Node, Literal
import os
class SqliteECB(EntityCentricKnowledgeGraphBackend):
    """
    SQLite backend for the knowledge graph.
    """
    
    def __init__(self, db_url: str):
        """
        Initialize the SQLite backend.
        Create the database file if it doesn't exist.
        """
        db_path = db_url.split("sqlite://")[-1]
        if not os.path.exists(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uri TEXT UNIQUE
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                subject INTEGER,
                predicate INTEGER,
                object INTEGER,
                FOREIGN KEY(subject) REFERENCES entities(id),
                FOREIGN KEY(predicate) REFERENCES entities(id),
                FOREIGN KEY(object) REFERENCES entities(id)
            )
        """)

        # The UNIQUE constraint on uri already creates an implicit unique index,
        # but if you want an explicit one, you can still add this:
        self.cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_uri ON entities (uri)
        """)

        self.cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_triples_spo ON triples(subject, predicate, object);        
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
        """)

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
        """)

        self.conn.commit()

    # assume: self.conn is an sqlite3.Connection and self.cursor = self.conn.cursor()
    # tip: right after opening the connection, do: self.conn.execute("PRAGMA foreign_keys = ON")

    def __close(self):
        try:
            self.cursor.close()
        finally:
            self.conn.close()

    def __execute(self, query: str, params: tuple | None = None):
        self.cursor.execute(query, params or ())

    def __executemany(self, query: str, seq_of_params: list[tuple]):
        self.cursor.executemany(query, seq_of_params)

    def _entity_id_for_term(self, term) -> int:
        if isinstance(term, Node):
            key = term.uri
        else:  # Literal
            key = f"__lit__:{term.value}"
        # get_or_create with proper params
        self.__execute("INSERT OR IGNORE INTO entities (uri) VALUES (?)", (key,))
        self.__execute("SELECT id FROM entities WHERE uri = ?", (key,))
        return self.cursor.fetchone()[0]

    def add_triples(self, triples: list[Triple]):
        # Convert objects → tuples of ints for executemany
        rows: list[tuple[int,int,int]] = []
        for t in triples:
            s_id = self._entity_id_for_term(t.subject)
            p_id = self._entity_id_for_term(t.predicate)
            o_id = self._entity_id_for_term(t.object)
            rows.append((s_id, p_id, o_id))

        self.__executemany(
            "INSERT INTO triples (subject, predicate, object) VALUES (?, ?, ?)",
            rows
        )
        self.conn.commit()

    def get_triples_by_subject_uri(self, subject_uri: str):
        # look up subject id then query
        self.__execute("SELECT id FROM entities WHERE uri = ?", (subject_uri,))
        row = self.cursor.fetchone()
        if not row:
            return []
        s_id = row[0]
        self.__execute("SELECT subject, predicate, object FROM triples WHERE subject = ?", (s_id,))
        return self.cursor.fetchall()

    def get_all_triples(self) -> list[tuple[str, str, str]]:
        query = """
            SELECT 
                s.uri AS subject_uri,
                p.uri AS predicate_uri,
                o.uri AS object_uri
            FROM triples t
            JOIN entities s ON t.subject = s.id
            JOIN entities p ON t.predicate = p.id
            JOIN entities o ON t.object = o.id
        """
        self.__execute(query)
        rows = self.cursor.fetchall()

        triples = []
        for s_uri, p_uri, o_uri in rows:
            # Decode literal placeholders if using the "__lit__:" convention
            if o_uri.startswith("__lit__:"):
                o_value = o_uri[len("__lit__:"):]
            else:
                o_value = o_uri
            triples.append((s_uri, p_uri, o_value))

        return triples

    def clear_database(self):
        self.__execute("DELETE FROM triples")
        self.__execute("DELETE FROM entities")
        self.conn.commit()