from kgpipe.common import KG, Data, DataFormat, Registry
from typing import Dict
import os
import pandas as pd
from rdflib import Graph, Literal, RDF
from collections import defaultdict



def __rdf_to_wide_csv(g: Graph, output_csv_path: str):


    # Dictionary to hold subject-based rows
    data = defaultdict(dict)

    # Track all unique predicate types with suffixes
    all_columns = set()

    for s, p, o in g:
        key = f"{p}_literal" if isinstance(o, Literal) else f"{p}_uri"
        all_columns.add(key)

        if key in data[s]:
            # If already exists, make it a list
            if isinstance(data[s][key], list):
                data[s][key].append(str(o))
            else:
                data[s][key] = [data[s][key], str(o)]
        else:
            data[s][key] = str(o)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    # Ensure all columns are included (even if some subjects lack values for them)
    for col in all_columns:
        if col not in df.columns:
            df[col] = None

    # Flatten lists into semicolon-separated strings
    for col in df.columns:
        df[col] = df[col].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)

    # Reset index to get subject as a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'subject'}, inplace=True)
    print(df.columns)
    sortkey = 'http://www.w3.org/2000/01/rdf-schema#label_literal'
    if sortkey not in df.columns:
        sortkey = 'subject'
    df.sort_values(by=sortkey, inplace=True, ascending=True)

    df.to_csv(output_csv_path, index=False)
    print(f"CSV written to {output_csv_path}")


def __rdf_to_wide_csv_v2(rdf_file_path: str, output_csv_path: str):


    def get_types(graph: Graph):
        types = set()
        for s, p, o in graph:
            if p == RDF.type:
                types.add(str(o))
        return types

    def get_subjects_by_type(graph: Graph, type: str):
        subjects = set()
        for s, p, o in graph:
            if p == RDF.type and str(o) == type:
                subjects.add(str(s))
        return subjects

    def filter_graph_by_subjects(graph: Graph, subjects: set):
        new_graph = Graph()
        for s, p, o in graph:
            if str(s) in subjects:
                new_graph.add((s, p, o))
        return new_graph

    g = Graph()
    if os.path.isdir(rdf_file_path):
        for file in os.listdir(rdf_file_path):
            g.parse(os.path.join(rdf_file_path,file))
    else:
        g.parse(rdf_file_path)

    # make dir output_csv_path
    os.makedirs(output_csv_path, exist_ok=True)

    for ty in get_types(g):
        subjects = get_subjects_by_type(g, ty)
        fg = filter_graph_by_subjects(g, subjects)
        fname = f"{'_'.join(ty.split('/')[3:])}.csv"
        __rdf_to_wide_csv(fg, f"{output_csv_path}/{fname}")

@Registry.task(
    input_spec={"rdf1": DataFormat.RDF_NTRIPLES, "rdf2": DataFormat.RDF_NTRIPLES},
    output_spec={"csv1": DataFormat.CSV, "csv2": DataFormat.CSV},
    description="Transform two RDF files to two CSV files",
    category=["Transform"]
)
def transform2_rdf_to_csv(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    __rdf_to_wide_csv(str(inputs["rdf1"].path), str(outputs["csv1"].path))
    __rdf_to_wide_csv(str(inputs["rdf2"].path), str(outputs["csv2"].path))

@Registry.task(
    input_spec={"rdf": DataFormat.RDF_NTRIPLES},
    output_spec={"csv": DataFormat.CSV},
    description="Transform an RDF file to a CSV file",
    category=["Transform"]
)
def transform_rdf_to_csv(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    g = Graph()

    rdf_file_path = str(inputs["rdf"].path)

    if os.path.isdir(rdf_file_path):
        for file in os.listdir(rdf_file_path):
            g.parse(os.path.join(rdf_file_path,file))
    else:
        g.parse(rdf_file_path)

    __rdf_to_wide_csv(g, str(outputs["csv"].path))


@Registry.task(
    input_spec={"rdf1": DataFormat.RDF_NTRIPLES, "rdf2": DataFormat.RDF_NTRIPLES},
    output_spec={"csv1": DataFormat.CSV, "csv2": DataFormat.CSV},
    description="Transform an RDF file to a CSV file",
    category=["Transform"]
)
def transform2_rdf_to_csv_v2(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    __rdf_to_wide_csv_v2(str(inputs["rdf1"].path), str(outputs["csv1"].path))
    __rdf_to_wide_csv_v2(str(inputs["rdf2"].path), str(outputs["csv2"].path))