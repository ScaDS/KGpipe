from turtle import back
from streamlit import table, title, text_input, button, write
import streamlit as st
from kgpipe.common.registry import Registry
import kgpipe_tasks.tasks
import sqlite3
import graphviz

title("KGpipe View")
st.set_page_config(layout="wide")

from streamlit_cytoscapejs import st_cytoscapejs

elements = [
    {"data": {"id": "one", "label": "Node 1"}, "position": {"x": 0, "y": 0}},
    {"data": {"id": "two", "label": "Node 2"}, "position": {"x": 100, "y": 0}},
    {"data": {"source": "one", "target": "two", "label": "Edge from Node1 to Node2"}},
]
stylesheet = [
    {"selector": "node", "style": {"width": 20, "height": 20, "shape": "rectangle"}},
    {"selector": "edge", "style": {"width": 10}},
]

clicked_elements = st_cytoscapejs(elements, stylesheet, width=1000, height=1000)

if clicked_elements is not None:
    st.write(clicked_elements)

# # wide streamlit view
# wide_view = True

# from kgpipe.common.systemgraph import backend

# def sparql(query: str):
#     qr = backend.query_sparql(query)
#     bindings = qr["results"]["bindings"]
#     results = []
#     for binding in bindings:
#         keys = binding.keys()
#         row = {}
#         for key in keys:
#             row[key] = binding[key]["value"]
#         results.append(row)
#     return results

# # create sqlite3 database
# conn = sqlite3.connect("kgpipe_view.db")
# cursor = conn.cursor()
# cursor.execute("CREATE TABLE IF NOT EXISTS queries (id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT)")
# conn.commit()

# def save_query(query: str):
#     cursor.execute("INSERT INTO queries (query) VALUES (?)", (query,))
#     conn.commit()

# def get_queries():
#     cursor.execute("SELECT * FROM queries")
#     return cursor.fetchall()

# queries = get_queries()
# # drop down menu for queries
# query_dropdown = st.selectbox("Queries", [q[1] for q in queries])

# # query field
# query = text_input("SELECT * { ?s ?p ?o . } LIMIT 10", value=query_dropdown)
# if button("Execute"):
#     query_result = sparql(query)
#     table(query_result)

# # save query button
# if button("Save Query"):
#     save_query(query)
#     queries = get_queries()
#     query_dropdown = st.selectbox("Queries", [q[1] for q in queries])


# def graph_visualization(query_result: list):
#     graph = graphviz.Digraph()
#     for row in query_result:
#         graph.edge(row["s"], row["o"])
#     return graph

# # graph visualization
# graph = graph_visualization(query_result)
# st.graphviz_chart(graph)