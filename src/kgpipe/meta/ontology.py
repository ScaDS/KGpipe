# """
# KGbench Ontology

# This module defines the core ontology for representing KGbench metadata
# as a structured knowledge graph, including tasks, pipelines, executions,
# data, metrics, and provenance information.
# """

# from typing import Dict, List, Any, Optional
# from dataclasses import dataclass, field
# from pathlib import Path
# import uuid
# from datetime import datetime

# # KGbench namespace and ontology IRI
# KGBENCH_NAMESPACE = "http://kgpipe.org/ontology#"
# KGBENCH_ONTOLOGY = "http://kgpipe.org/ontology"

# # Common namespaces
# RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
# RDFS_NAMESPACE = "http://www.w3.org/2000/01/rdf-schema#"
# OWL_NAMESPACE = "http://www.w3.org/2002/07/owl#"
# XSD_NAMESPACE = "http://www.w3.org/2001/XMLSchema#"
# PROV_NAMESPACE = "http://www.w3.org/ns/prov#"
# DCTERMS_NAMESPACE = "http://purl.org/dc/terms/"
# SCHEMA_NAMESPACE = "http://schema.org/"
# DUBLIN_CORE_NAMESPACE = "http://purl.org/dc/elements/1.1/"


# # Core KGbench Classes
# class TaskClass(OntologyClass):
#     """Task class representing executable components."""
    
#     def __init__(self):
#         super().__init__(
#             iri=f"{KGBENCH_NAMESPACE}Task",
#             label="Task",
#             description="An executable component in a knowledge graph pipeline",
#             superclasses=[f"{PROV_NAMESPACE}Activity"],
#             properties=[
#                 f"{KGBENCH_NAMESPACE}hasInputSpec",
#                 f"{KGBENCH_NAMESPACE}hasOutputSpec", 
#                 f"{KGBENCH_NAMESPACE}hasCategory",
#                 f"{KGBENCH_NAMESPACE}hasImplementation",
#                 f"{KGBENCH_NAMESPACE}hasVersion",
#                 f"{KGBENCH_NAMESPACE}hasDockerImage",
#                 f"{KGBENCH_NAMESPACE}hasCommand",
#                 f"{PROV_NAMESPACE}startedAtTime",
#                 f"{PROV_NAMESPACE}endedAtTime",
#                 f"{PROV_NAMESPACE}wasAssociatedWith"
#             ]
#         )


# class PipelineClass(OntologyClass):
#     """Pipeline class representing workflow definitions."""
    
#     def __init__(self):
#         super().__init__(
#             iri=f"{KGBENCH_NAMESPACE}Pipeline",
#             label="Pipeline", 
#             description="A workflow definition consisting of multiple tasks",
#             superclasses=[f"{PROV_NAMESPACE}Plan"],
#             properties=[
#                 f"{KGBENCH_NAMESPACE}hasTask",
#                 f"{KGBENCH_NAMESPACE}hasStage",
#                 f"{KGBENCH_NAMESPACE}hasTarget",
#                 f"{KGBENCH_NAMESPACE}hasConfiguration",
#                 f"{KGBENCH_NAMESPACE}hasVersion",
#                 f"{PROV_NAMESPACE}wasGeneratedBy",
#                 f"{PROV_NAMESPACE}wasDerivedFrom"
#             ]
#         )


# class ExecutionClass(OntologyClass):
#     """Execution class representing pipeline runs."""
    
#     def __init__(self):
#         super().__init__(
#             iri=f"{KGBENCH_NAMESPACE}Execution",
#             label="Execution",
#             description="A specific execution of a pipeline",
#             superclasses=[f"{PROV_NAMESPACE}Activity"],
#             properties=[
#                 f"{PROV_NAMESPACE}used",
#                 f"{PROV_NAMESPACE}generated",
#                 f"{PROV_NAMESPACE}wasAssociatedWith",
#                 f"{PROV_NAMESPACE}startedAtTime",
#                 f"{PROV_NAMESPACE}endedAtTime",
#                 f"{KGBENCH_NAMESPACE}hasStatus",
#                 f"{KGBENCH_NAMESPACE}hasExitCode",
#                 f"{KGBENCH_NAMESPACE}hasLogs",
#                 f"{KGBENCH_NAMESPACE}hasMetrics"
#             ]
#         )


# class DataClass(OntologyClass):
#     """Data class representing input/output data."""
    
#     def __init__(self):
#         super().__init__(
#             iri=f"{KGBENCH_NAMESPACE}Data",
#             label="Data",
#             description="Input or output data in a pipeline",
#             superclasses=[f"{PROV_NAMESPACE}Entity"],
#             properties=[
#                 f"{KGBENCH_NAMESPACE}hasFormat",
#                 f"{KGBENCH_NAMESPACE}hasPath",
#                 f"{KGBENCH_NAMESPACE}hasSize",
#                 f"{KGBENCH_NAMESPACE}hasChecksum",
#                 f"{KGBENCH_NAMESPACE}hasMetadata",
#                 f"{PROV_NAMESPACE}wasGeneratedBy",
#                 f"{PROV_NAMESPACE}wasDerivedFrom",
#                 f"{PROV_NAMESPACE}wasAttributedTo"
#             ]
#         )


# class MetricClass(OntologyClass):
#     """Metric class representing performance and quality metrics."""
    
#     def __init__(self):
#         super().__init__(
#             iri=f"{KGBENCH_NAMESPACE}Metric",
#             label="Metric",
#             description="A performance or quality metric",
#             superclasses=[f"{PROV_NAMESPACE}Entity"],
#             properties=[
#                 f"{KGBENCH_NAMESPACE}hasType",
#                 f"{KGBENCH_NAMESPACE}hasValue",
#                 f"{KGBENCH_NAMESPACE}hasUnit",
#                 f"{KGBENCH_NAMESPACE}hasThreshold",
#                 f"{PROV_NAMESPACE}wasGeneratedBy",
#                 f"{PROV_NAMESPACE}wasAttributedTo"
#             ]
#         )


# class ProvenanceClass(OntologyClass):
#     """Provenance class representing lineage information."""
    
#     def __init__(self):
#         super().__init__(
#             iri=f"{KGBENCH_NAMESPACE}Provenance",
#             label="Provenance",
#             description="Provenance information for data and executions",
#             superclasses=[f"{PROV_NAMESPACE}Entity"],
#             properties=[
#                 f"{PROV_NAMESPACE}wasGeneratedBy",
#                 f"{PROV_NAMESPACE}wasDerivedFrom",
#                 f"{PROV_NAMESPACE}wasAttributedTo",
#                 f"{PROV_NAMESPACE}wasAssociatedWith",
#                 f"{PROV_NAMESPACE}startedAtTime",
#                 f"{PROV_NAMESPACE}endedAtTime",
#                 f"{KGBENCH_NAMESPACE}hasLineage",
#                 f"{KGBENCH_NAMESPACE}hasConfidence"
#             ]
#         )


# # Core KGbench Properties
# KGBENCH_PROPERTIES = {
#     # Task properties
#     "hasInputSpec": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasInputSpec",
#         label="has input specification",
#         domain=f"{KGBENCH_NAMESPACE}Task",
#         range=f"{KGBENCH_NAMESPACE}InputSpec",
#         description="Specifies the input requirements for a task"
#     ),
#     "hasOutputSpec": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasOutputSpec", 
#         label="has output specification",
#         domain=f"{KGBENCH_NAMESPACE}Task",
#         range=f"{KGBENCH_NAMESPACE}OutputSpec",
#         description="Specifies the output format for a task"
#     ),
#     "hasCategory": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasCategory",
#         label="has category",
#         domain=f"{KGBENCH_NAMESPACE}Task",
#         range=f"{KGBENCH_NAMESPACE}TaskCategory",
#         description="The category of a task (e.g., entity linking, information extraction)"
#     ),
#     "hasImplementation": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasImplementation",
#         label="has implementation",
#         domain=f"{KGBENCH_NAMESPACE}Task",
#         range=f"{XSD_NAMESPACE}string",
#         description="The implementation details of a task"
#     ),
#     "hasDockerImage": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasDockerImage",
#         label="has Docker image",
#         domain=f"{KGBENCH_NAMESPACE}Task",
#         range=f"{XSD_NAMESPACE}string",
#         description="The Docker image used by a task"
#     ),
#     "hasCommand": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasCommand",
#         label="has command",
#         domain=f"{KGBENCH_NAMESPACE}Task",
#         range=f"{XSD_NAMESPACE}string",
#         description="The command executed by a task"
#     ),
    
#     # Pipeline properties
#     "hasTask": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasTask",
#         label="has task",
#         domain=f"{KGBENCH_NAMESPACE}Pipeline",
#         range=f"{KGBENCH_NAMESPACE}Task",
#         description="A task that is part of a pipeline"
#     ),
#     "hasStage": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasStage",
#         label="has stage",
#         domain=f"{KGBENCH_NAMESPACE}Pipeline",
#         range=f"{KGBENCH_NAMESPACE}Stage",
#         description="A stage that is part of a pipeline"
#     ),
#     "hasTarget": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasTarget",
#         label="has target",
#         domain=f"{KGBENCH_NAMESPACE}Pipeline",
#         range=f"{KGBENCH_NAMESPACE}Data",
#         description="The target output of a pipeline"
#     ),
#     "hasConfiguration": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasConfiguration",
#         label="has configuration",
#         domain=f"{KGBENCH_NAMESPACE}Pipeline",
#         range=f"{KGBENCH_NAMESPACE}Configuration",
#         description="The configuration parameters for a pipeline"
#     ),
    
#     # Execution properties
#     "hasStatus": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasStatus",
#         label="has status",
#         domain=f"{KGBENCH_NAMESPACE}Execution",
#         range=f"{KGBENCH_NAMESPACE}ExecutionStatus",
#         description="The status of an execution"
#     ),
#     "hasExitCode": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasExitCode",
#         label="has exit code",
#         domain=f"{KGBENCH_NAMESPACE}Execution",
#         range=f"{XSD_NAMESPACE}integer",
#         description="The exit code of an execution"
#     ),
#     "hasLogs": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasLogs",
#         label="has logs",
#         domain=f"{KGBENCH_NAMESPACE}Execution",
#         range=f"{XSD_NAMESPACE}string",
#         description="The log output of an execution"
#     ),
#     "hasMetrics": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasMetrics",
#         label="has metrics",
#         domain=f"{KGBENCH_NAMESPACE}Execution",
#         range=f"{KGBENCH_NAMESPACE}Metric",
#         description="Metrics associated with an execution"
#     ),
    
#     # Data properties
#     "hasFormat": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasFormat",
#         label="has format",
#         domain=f"{KGBENCH_NAMESPACE}Data",
#         range=f"{KGBENCH_NAMESPACE}DataFormat",
#         description="The format of a data entity"
#     ),
#     "hasPath": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasPath",
#         label="has path",
#         domain=f"{KGBENCH_NAMESPACE}Data",
#         range=f"{XSD_NAMESPACE}string",
#         description="The file path of a data entity"
#     ),
#     "hasSize": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasSize",
#         label="has size",
#         domain=f"{KGBENCH_NAMESPACE}Data",
#         range=f"{XSD_NAMESPACE}long",
#         description="The size of a data entity in bytes"
#     ),
#     "hasChecksum": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasChecksum",
#         label="has checksum",
#         domain=f"{KGBENCH_NAMESPACE}Data",
#         range=f"{XSD_NAMESPACE}string",
#         description="The checksum of a data entity"
#     ),
#     "hasMetadata": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasMetadata",
#         label="has metadata",
#         domain=f"{KGBENCH_NAMESPACE}Data",
#         range=f"{KGBENCH_NAMESPACE}Metadata",
#         description="Additional metadata for a data entity"
#     ),
    
#     # Metric properties
#     "hasType": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasType",
#         label="has type",
#         domain=f"{KGBENCH_NAMESPACE}Metric",
#         range=f"{KGBENCH_NAMESPACE}MetricType",
#         description="The type of a metric"
#     ),
#     "hasValue": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasValue",
#         label="has value",
#         domain=f"{KGBENCH_NAMESPACE}Metric",
#         range=f"{XSD_NAMESPACE}double",
#         description="The value of a metric"
#     ),
#     "hasUnit": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasUnit",
#         label="has unit",
#         domain=f"{KGBENCH_NAMESPACE}Metric",
#         range=f"{XSD_NAMESPACE}string",
#         description="The unit of a metric"
#     ),
#     "hasThreshold": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasThreshold",
#         label="has threshold",
#         domain=f"{KGBENCH_NAMESPACE}Metric",
#         range=f"{XSD_NAMESPACE}double",
#         description="The threshold value for a metric"
#     ),
    
#     # Provenance properties
#     "hasLineage": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasLineage",
#         label="has lineage",
#         domain=f"{KGBENCH_NAMESPACE}Provenance",
#         range=f"{KGBENCH_NAMESPACE}Lineage",
#         description="The lineage information for provenance"
#     ),
#     "hasConfidence": OntologyProperty(
#         iri=f"{KGBENCH_NAMESPACE}hasConfidence",
#         label="has confidence",
#         domain=f"{KGBENCH_NAMESPACE}Provenance",
#         range=f"{XSD_NAMESPACE}double",
#         description="The confidence level of provenance information"
#     )
# }


# # Complete ontology definition
# KGBENCH_ONTOLOGY = {
#     "namespace": KGBENCH_NAMESPACE,
#     "iri": KGBENCH_ONTOLOGY,
#     "title": "KGbench Ontology",
#     "description": "Ontology for representing KGbench metadata as structured knowledge graphs",
#     "version": "1.0.0",
#     "classes": {
#         "Task": TaskClass(),
#         "Pipeline": PipelineClass(),
#         "Execution": ExecutionClass(),
#         "Data": DataClass(),
#         "Metric": MetricClass(),
#         "Provenance": ProvenanceClass()
#     },
#     "properties": KGBENCH_PROPERTIES,
#     "namespaces": {
#         "kgpipe": KGBENCH_NAMESPACE,
#         "rdf": RDF_NAMESPACE,
#         "rdfs": RDFS_NAMESPACE,
#         "owl": OWL_NAMESPACE,
#         "xsd": XSD_NAMESPACE,
#         "prov": PROV_NAMESPACE,
#         "dcterms": DCTERMS_NAMESPACE,
#         "schema": SCHEMA_NAMESPACE,
#         "dc": DUBLIN_CORE_NAMESPACE
#     }
# } 