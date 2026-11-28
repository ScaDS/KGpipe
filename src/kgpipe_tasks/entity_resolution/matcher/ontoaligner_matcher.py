# https://github.com/sciknoworg/OntoAligner

from pathlib import Path
from typing import Dict, Any
from kgpipe.common.model.configuration import ConfigurationDefinition, Parameter, ParameterType
import pandas as pd
import os
import csv


from kgpipe.common import KgTask, DataFormat, Data, Registry, BasicTaskCategoryCatalog
from kgpipe.common.io import get_docker_volume_bindings, remap_data_path_for_container
from kgpipe.execution import docker_client
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Match, ER_Document

# ALl matcher variants:
# Fuzzy Matching
# Retrieval Aligner
# LLM Aligner
# RAG Aligner
# FewShot-RAG Aligner
# In-Context Vectors RAG
# KGE Aligner
# PropMatch Aligner
# FLORA Aligner
# OLaLa Aligner
# Ensemble Learning
# Product Alignment in eCommerce
# FIBO Corporate Actions Alignment

common_spec = ConfigurationDefinition(
        name="OntoAligner schema alignment configuration",
        description="OntoAligner schema alignment configuration",
        parameters=[
            Parameter(
                name="todo",
                description="needs to be implemented",
                datatype=ParameterType.string,
                default_value="none",
            ),
        ]
    )

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner Fuzzy Matching entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_fuzzy_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner Fuzzy Matching entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner Retrieval Aligner entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_retrieval_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner Retrieval Aligner entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner LLM Aligner entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_llm_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner LLM Aligner entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner RAG Aligner entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_rag_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner RAG Aligner entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner FewShot-RAG Aligner entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_fewshot_rag_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner FewShot-RAG Aligner entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner In-Context Vectors RAG Aligner entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_in_context_vectors_rag_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner In-Context Vectors RAG Aligner entity matching using Docker container
    """
    pass
    
@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner KGE Aligner entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_kge_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner KGE Aligner entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner PropMatch Aligner entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_propmatch_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner PropMatch Aligner entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner FLORA Aligner entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_flora_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner FLORA Aligner entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner OLaLa Aligner entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_olala_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner OLaLa Aligner entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner Ensemble Learning entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_ensemble_learning_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner Ensemble Learning entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner Product Alignment in eCommerce entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_product_alignment_in_ecommerce_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner Product Alignment in eCommerce entity matching using Docker container
    """
    pass

@Registry.task(
    input_spec={"source": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    description="OntoAligner FIBO Corporate Actions Alignment entity matching using Docker container",
    category=BasicTaskCategoryCatalog.schema_alignment,
    see_also=["https://github.com/sciknoworg/OntoAligner"],
    config_spec=common_spec,
)
def ontoaligner_fibo_corporate_actions_alignment_matcher(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    OntoAligner FIBO Corporate Actions Alignment entity matching using Docker container
    """
    pass