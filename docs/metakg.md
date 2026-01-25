# Meta Knowledge Graph (PipeKG)

The KGpipe framework maintains an internal meta knowledge graph (PipeKG) that serves as a knowledge base for the framework itself. This meta KG tracks tasks, tool implementations, their components, pipelines, metrics, and evaluation results.

The PipeKG enables several key capabilities:

- **Defining Tasks**: Tasks are automatically registered in the meta KG when defined using the `@Registry.task` decorator
- **Tracking Pipelines**: Pipeline definitions and execution results are stored in the meta KG
- **Generating Viable Pipelines**: The automatic pipeline generation algorithm queries the meta KG to find compatible task sequences
- **Tracking KG Evaluation**: Evaluation metrics and results are recorded in the meta KG for analysis

## Querying the Meta KG

The meta KG can be queried using SPARQL to discover available tasks, pipelines, and their relationships. Here are some useful queries:

### Generate Simple Ontology

Extract the ontology structure from the meta KG:

```sparql
CONSTRUCT {
  ?s ?p ?o .
} WHERE {
  ?s a ?t .
  ?s ?p ?o .
  FILTER(strstarts(str(?t),'http://www.w3.org/2002/07/owl#'))
}
```

### List All Tasks

Get a comprehensive list of all registered tasks with their categories, descriptions, and input/output formats:

```sparql
SELECT 
  ?task 
  (group_concat(distinct ?cat;separator=",\n") as ?cats) 
  ?desc 
  (group_concat(distinct ?ind;separator=",\n") as ?inds)  
  (group_concat(distinct ?inf;separator=",\n") as ?infs) 
  (group_concat(distinct ?outd;separator=",\n") as ?outds)  
  (group_concat(distinct ?outf;separator=",\n") as ?outfs) 
WHERE { 
    ?task a <Task> . 
    ?task a ?cat .
    ?task <description> ?desc .
    ?task <input> ?ind .
    ?task <input> ?outd .
    ?ind a <Data> .
    ?ind <format> ?inf .
    ?outd a <Data> .
    ?outd <format> ?outf .
}
GROUP BY ?task ?desc
```

This query returns:
- `?task`: The task identifier
- `?cats`: Categories the task belongs to
- `?desc`: Task description
- `?inds`: Input data identifiers
- `?infs`: Input formats
- `?outds`: Output data identifiers
- `?outfs`: Output formats

## Accessing the Meta KG

The meta KG is accessible through the `SYS_KG` object. You can query it programmatically or through SPARQL endpoints depending on the storage backend configuration (see [Configuration](configuration.md)).
