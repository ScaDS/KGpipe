import importlib.util
from pathlib import Path


def _load_converter_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "kgpipe_view" / "owl_to_mermaid.py"
    )
    spec = importlib.util.spec_from_file_location("owl_to_mermaid", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_get_available_layers_includes_core_layer():
    module = _load_converter_module()
    ttl_path = Path(__file__).resolve().parents[2] / "kgpipe_view" / "kgpipe.owl.ttl"

    layers = module.get_available_layers(ttl_path)

    assert "CoreLayer" in layers
    assert "PipelineLayer" in layers


def test_convert_owl_ttl_to_mermaid_filters_by_layer():
    module = _load_converter_module()
    ttl_path = Path(__file__).resolve().parents[2] / "kgpipe_view" / "kgpipe.owl.ttl"

    mermaid = module.convert_owl_ttl_to_mermaid(ttl_path, layer_filter="CoreLayer")

    assert "class Task" in mermaid
    assert "class Method" in mermaid
    assert "class Pipeline" not in mermaid
    assert "class PipelineStep" not in mermaid
    assert 'Method "0..*" --> "0..*" Task : realizesTask' in mermaid
    assert 'Pipeline "0..*" --> "0..*" PipelineStep : hasStep' not in mermaid


def test_convert_owl_ttl_to_mermaid_filters_by_multiple_layers():
    module = _load_converter_module()
    ttl_path = Path(__file__).resolve().parents[2] / "kgpipe_view" / "kgpipe.owl.ttl"

    mermaid = module.convert_owl_ttl_to_mermaid(
        ttl_path, layer_filter=["CoreLayer", "PipelineLayer"]
    )

    assert "class Task" in mermaid
    assert "class Pipeline" in mermaid
    assert 'Pipeline "0..*" --> "0..*" PipelineStep : hasStep' in mermaid
    assert "class Artifact" not in mermaid
