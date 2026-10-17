from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

ML_PACKAGES = ("torch", "transformers", "sentence_transformers")
SRC_ROOT = Path(__file__).resolve().parents[2]


def test_no_top_level_ml_imports_in_src() -> None:
    """Ensure no module under src/ imports torch/transformers/sentence_transformers at top level."""
    offending: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue

            for mod in modules:
                if any(mod == pkg or mod.startswith(f"{pkg}.") for pkg in ML_PACKAGES):
                    offending.append(f"{path.relative_to(SRC_ROOT)}:{node.lineno} ({mod})")

    assert offending == []


def test_embeddings_import_does_not_load_ml_dependencies(monkeypatch) -> None:
    """Importing kgpipe.util.embeddings and st_emb should not require torch or sentence_transformers."""
    for pkg in (*ML_PACKAGES, "kgpipe.util.embeddings", "kgpipe.util.embeddings.st_emb"):
        monkeypatch.delitem(sys.modules, pkg, raising=False)
    for pkg in ML_PACKAGES:
        monkeypatch.setitem(sys.modules, pkg, None)

    embeddings = importlib.import_module("kgpipe.util.embeddings")
    st_emb = importlib.import_module("kgpipe.util.embeddings.st_emb")

    assert hasattr(embeddings, "get_embedder")
    assert hasattr(st_emb, "SentenceTransformerEmbedder")


def test_st_emb_get_model_lazy_loads_sentence_transformer(monkeypatch) -> None:
    """get_model() lazily imports SentenceTransformer and torch on first encode call."""
    import kgpipe.util.embeddings.st_emb as st_emb

    fake_vecs = np.ones((2, 4), dtype=np.float32)
    fake_model = MagicMock()
    fake_model.encode.return_value = fake_vecs

    fake_st_cls = MagicMock(return_value=fake_model)
    fake_st_mod = MagicMock(SentenceTransformer=fake_st_cls)
    fake_torch_mod = MagicMock()
    fake_torch_mod.cuda.is_available.return_value = False

    monkeypatch.setattr(st_emb, "model", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st_mod)
    monkeypatch.setitem(sys.modules, "torch", fake_torch_mod)

    embedder = st_emb.SentenceTransformerEmbedder()
    fake_st_cls.assert_not_called()

    result = embedder.encode(["a", "b"])
    fake_st_cls.assert_called_once_with("all-MiniLM-L6-v2")
    assert np.array_equal(result, fake_vecs)
