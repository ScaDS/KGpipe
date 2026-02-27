from __future__ import annotations

from typing import Any

import pandas as pd


PRIMARY_QUERY = """
PREFIX kgp: <http://github.com/ScaDS/kgpipe/ontology/>

SELECT ?task ?method ?implementation ?tool ?runtime ?implementationVersion ?commandTemplate
WHERE {
  ?implementation a kgp:Implementation .
  OPTIONAL { ?implementation kgp:implementsMethod ?method . }
  OPTIONAL { ?implementation kgp:usesTool ?tool . }
  OPTIONAL { ?implementation kgp:runtime ?runtime . }
  OPTIONAL { ?implementation kgp:implementationVersion ?implementationVersion . }
  OPTIONAL { ?implementation kgp:commandTemplate ?commandTemplate . }
  OPTIONAL { ?method kgp:realizesTask ?task . }
}
ORDER BY ?task ?implementation
"""


TASK_HIERARCHY_PRIMARY_QUERY = """
PREFIX kgp: <http://github.com/ScaDS/kgpipe/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT DISTINCT ?task ?parentTask
WHERE {
  {
    ?task a kgp:Task .
  }
  UNION
  {
    ?task a owl:Class .
    ?task rdfs:subClassOf+ kgp:Task .
    FILTER(?task != kgp:Task)
  }
  UNION
  {
    ?method kgp:realizesTask ?task .
  }
  FILTER(isIRI(?task))
  OPTIONAL {
    ?task rdfs:subClassOf ?parentTask .
    ?parentTask rdfs:subClassOf* kgp:Task .
    FILTER(?parentTask != owl:Thing)
  }
}
ORDER BY ?task ?parentTask
"""


TASK_HIERARCHY_FALLBACK_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?task ?parentTask
WHERE {
  {
    ?task a ?taskType .
    FILTER(STRENDS(STR(?taskType), "Task"))
  }
  UNION
  {
    ?task a ?classType .
    FILTER(STRENDS(STR(?classType), "Class"))
    ?task rdfs:subClassOf+ ?taskRoot .
    FILTER(STRENDS(STR(?taskRoot), "Task"))
    FILTER(?task != ?taskRoot)
  }
  UNION
  {
    ?method ?realizesTaskPredicate ?task .
    FILTER(STRENDS(STR(?realizesTaskPredicate), "realizesTask"))
  }
  FILTER(isIRI(?task))
  OPTIONAL {
    ?task rdfs:subClassOf ?parentTask .
    FILTER(STRENDS(STR(?parentTask), "Task"))
  }
}
ORDER BY ?task ?parentTask
"""


FALLBACK_QUERY = """
SELECT ?task ?method ?implementation ?tool ?runtime ?implementationVersion ?commandTemplate
WHERE {
  ?implementation a ?implementationType .
  FILTER(STRENDS(STR(?implementationType), "Implementation"))

  OPTIONAL {
    ?implementation ?implementsMethodPredicate ?method .
    FILTER(STRENDS(STR(?implementsMethodPredicate), "implementsMethod"))
  }
  OPTIONAL {
    ?method ?realizesTaskPredicate ?task .
    FILTER(STRENDS(STR(?realizesTaskPredicate), "realizesTask"))
  }
  OPTIONAL {
    ?implementation ?usesToolPredicate ?tool .
    FILTER(STRENDS(STR(?usesToolPredicate), "usesTool"))
  }
  OPTIONAL {
    ?implementation ?runtimePredicate ?runtime .
    FILTER(STRENDS(STR(?runtimePredicate), "runtime"))
  }
  OPTIONAL {
    ?implementation ?implementationVersionPredicate ?implementationVersion .
    FILTER(STRENDS(STR(?implementationVersionPredicate), "implementationVersion"))
  }
  OPTIONAL {
    ?implementation ?commandTemplatePredicate ?commandTemplate .
    FILTER(STRENDS(STR(?commandTemplatePredicate), "commandTemplate"))
  }
}
ORDER BY ?task ?implementation
"""

PIPELINE_RUN_QUERY = """
PREFIX kgp: <http://github.com/ScaDS/kgpipe/ontology/>

SELECT DISTINCT ?pipelineRun
WHERE {
  ?pipelineRun a kgp:PipelineRun .
}
"""

KG_DATA_QUERY = """
PREFIX kgp: <http://github.com/ScaDS/kgpipe/ontology/>

SELECT DISTINCT ?kgData
WHERE {
  VALUES ?format {
    ".nt"
    ".ttl"
    ".rdf"
    ".jsonld"
  }
  ?kgData a kgp:Data .
  ?kgData <format> ?format .
}
"""


def _run_select(endpoint_url: str, query: str) -> list[dict[str, Any]]:
    from SPARQLWrapper import JSON, SPARQLWrapper

    client = SPARQLWrapper(endpoint_url)
    client.setQuery(query)
    client.setReturnFormat(JSON)
    result = client.query().convert()
    return result.get("results", {}).get("bindings", [])


def _cell(binding: dict[str, Any], key: str) -> str:
    item = binding.get(key)
    if not item:
        return ""
    return str(item.get("value", ""))


def _to_task_rows(bindings: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for binding in bindings:
        rows.append(
            {
                "task": _cell(binding, "task"),
                "method": _cell(binding, "method"),
                "implementation": _cell(binding, "implementation"),
                "tool": _cell(binding, "tool"),
                "runtime": _cell(binding, "runtime"),
                "implementation_version": _cell(binding, "implementationVersion"),
                "command_template": _cell(binding, "commandTemplate"),
            }
        )
    return rows


def _to_task_hierarchy_rows(bindings: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for binding in bindings:
        rows.append(
            {
                "task": _cell(binding, "task"),
                "parent_task": _cell(binding, "parentTask"),
            }
        )
    return rows

def _to_pipeline_run_rows(bindings: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for binding in bindings:
        rows.append(
            {
                "pipeline_run": _cell(binding, "pipelineRun"),
            }
        )
    print(rows)
    return rows

def _to_kg_data_rows(bindings: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for binding in bindings:
        rows.append(
            {
                "kg_data": _cell(binding, "kgData"),
            }
        )
    return rows


def query_tasks_implementations(endpoint_url: str) -> pd.DataFrame:
    bindings = _run_select(endpoint_url, PRIMARY_QUERY)
    if not bindings:
        bindings = _run_select(endpoint_url, FALLBACK_QUERY)
    rows = _to_task_rows(bindings)
    return pd.DataFrame(rows)


def query_task_hierarchy(endpoint_url: str) -> pd.DataFrame:
    bindings = _run_select(endpoint_url, TASK_HIERARCHY_PRIMARY_QUERY)
    if not bindings:
        bindings = _run_select(endpoint_url, TASK_HIERARCHY_FALLBACK_QUERY)
    rows = _to_task_hierarchy_rows(bindings)
    return pd.DataFrame(rows)

def query_pipeline_hierarchy(endpoint_url: str) -> pd.DataFrame:
    bindings = _run_select(endpoint_url, PIPELINE_RUN_QUERY)
    rows = _to_pipeline_run_rows(bindings)
    return pd.DataFrame(rows)

def query_evaluation_hierarchy(endpoint_url: str) -> pd.DataFrame:
    # TODO: Implement evaluation hierarchy query
    # bindings = _run_select(endpoint_url, EVALUATION_HIERARCHY_QUERY)
    # rows = _to_evaluation_hierarchy_rows(bindings)
    # return pd.DataFrame(rows)
    return pd.DataFrame([])

def query_kg_data(endpoint_url: str) -> pd.DataFrame:
    bindings = _run_select(endpoint_url, KG_DATA_QUERY)
    rows = _to_kg_data_rows(bindings)
    return pd.DataFrame(rows)