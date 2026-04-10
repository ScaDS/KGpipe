from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from kgpipe.common import Data, DataFormat, KgPipe, KgTask  # pragma: no cover
    from kgpipe.common.model.task import KgTaskReport  # pragma: no cover

from .models import PipelineConfig, PipelineFamily


@dataclass(frozen=True)
class PipelineRunResult:
    family: PipelineFamily
    cfg: PipelineConfig
    workdir: Path
    final_output: Any  # Data
    task_reports: Any  # list[KgTaskReport]
    aux: dict


TEST_DATA_SEED_KG_PATH = Path("/home/marvin/project/data/final/film_1k/split_0/kg/seed/data.nt")
TEST_DATA_ONTOLOGY_PATH = Path("/home/marvin/project/data/final/film_1k/movie-ontology.ttl")
TEST_DATA_RDF_PATH = Path("/home/marvin/project/data/final/film_1k/split_1/sources/rdf/data.nt")
TEST_DATA_TEXT_PATH = Path("/home/marvin/project/data/final/film_1k/split_1/sources/text/data/")

def _import_tasks_for_family(family: PipelineFamily) -> None:
    """
    Import task modules so their @Registry.task decorators execute.

    This keeps the rest of qap_mock independent from kgpipe_tasks import side effects.
    """
    # RDF: PARIS matcher + exchange + fusion tasks.
    if family == PipelineFamily.RDF:
        # Entity matching (docker) + exchange (python)
        import kgpipe_tasks.entity_resolution.matcher.paris_rdf_matcher  # noqa: F401
        import kgpipe_tasks.entity_resolution.entity_match  # noqa: F401

        # Fusion (python)
        import kgpipe_tasks.entity_resolution.fusion.union  # noqa: F401
        import kgpipe_tasks.entity_resolution.fusion.preference  # noqa: F401

        return

    if family == PipelineFamily.TEXT:
        # CoreNLP OpenIE extraction (docker) + exchange (python)
        import kgpipe_tasks.text_processing.text_extraction.corenlp_extraction  # noqa: F401

        return

    raise ValueError(f"Unknown family: {family}")


def _cfg_hash(cfg: PipelineConfig) -> str:
    payload = json.dumps(cfg.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# def _test_data_path(relative_path: str) -> Path:
#     """
#     Use kgpipe_tasks' bundled test data as default inputs so qap_mock is runnable.
#     """
#     base = Path(__file__).resolve().parents[3] / "src" / "kgpipe_tasks" / "test" / "test_data"
#     path = (base / relative_path).resolve()
#     if not path.exists():
#         raise FileNotFoundError(f"Missing test data file: {path}")
#     return path


def _set_env_from_params(params: dict[str, float]) -> dict[str, Optional[str]]:
    """
    Apply a minimal mapping from qap_mock params to the env-var based configuration
    convention used by many kgpipe tasks.

    Returns a dict of previous env values so callers can restore them.
    """
    # Only set variables that are known to be read by the tasks we use.
    mapping: dict[str, tuple[str, float]] = {
        # RDF fusion/preference tasks
        "ENTITY_MATCHING_THRESHOLD": ("entity_sim_threshold", 0.7),
        "RELATION_MATCHING_THRESHOLD": ("schema_sim_threshold", 0.7),
        # Text: no stable env knobs used by CoreNLP task today
    }

    prev: dict[str, Optional[str]] = {}
    for env_key, (p_key, default) in mapping.items():
        prev[env_key] = os.environ.get(env_key)
        val = float(params.get(p_key, default))
        os.environ[env_key] = str(val)
    return prev


def _restore_env(prev: dict[str, Optional[str]]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def build_pipeline_for_config(*, cfg: PipelineConfig, workdir: Path) -> tuple[KgPipe, Data, Data]:
    """
    Build a runnable KgPipe for the given configuration.

    We intentionally keep the mapping small and explicit:
    - RDF: (optional) PARIS entity matching -> exchange -> fusion
    - TEXT: CoreNLP OpenIE extraction (docker) -> exchange

    Returns (pipe, source, final_result_data).
    """
    from kgpipe.common import Data, DataFormat, KgPipe, KgTask, Registry

    _import_tasks_for_family(cfg.family)
    _ensure_dir(workdir)

    if cfg.family == PipelineFamily.RDF:
        # Inputs: source + target (as seed) are bundled test fixtures.
        source = Data(path=TEST_DATA_RDF_PATH, format=DataFormat.RDF_NTRIPLES)
        target = Data(path=TEST_DATA_SEED_KG_PATH, format=DataFormat.RDF_NTRIPLES)

        # Ensure ontology env is set for fusion tasks that need it.
        ontology_path = TEST_DATA_ONTOLOGY_PATH
        os.environ.setdefault("ONTOLOGY_PATH", str(ontology_path))

        # Decide whether to run entity matching. If we don't, we can still
        # compute a meaningful output via simple union.
        entity_impl = cfg.implementations.get("entity_matching", "rule_based")
        fusion_impl = cfg.implementations.get("fusion", "union")
        use_docker = os.environ.get("QAP_MOCK_USE_DOCKER", "0") == "1"

        tasks: list[KgTask] = []
        final_format = DataFormat.RDF_NTRIPLES

        def _empty_er(inputs: dict[str, Data], outputs: dict[str, Data]) -> None:
            out_path = Path(outputs["output"].path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"matches": [], "blocks": [], "clusters": []}, indent=2), encoding="utf-8")

        # dummy_entity_matching = KgTask(
        #     name="dummy_entity_matching",
        #     input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES},
        #     output_spec={"output": DataFormat.ER_JSON},
        #     function=_empty_er,
        #     description="Dummy matcher emitting empty ER_JSON (no docker)",
        # )

        # if entity_impl != "rule_based":
        #     if use_docker:
        tasks.extend(
            [
                Registry.get_task("paris_entity_matching"),
                Registry.get_task("paris_exchange"),
            ]
        )
        # When matches exist, prefer a fusion strategy that uses them.
        if fusion_impl in ("quality_weighted", "majority_vote"):
            tasks.append(Registry.get_task("fusion_first_value"))
        else:
            tasks.append(Registry.get_task("union_matched_rdf"))
            # else:
            #     # Non-docker mode: skip PARIS and run a deterministic empty matcher.
            #     tasks.extend([dummy_entity_matching, Registry.get_task("union_matched_rdf")])
        # else:
        #     # No matching step: just union the two graphs.
        #     tasks.append(Registry.get_task("fusion_union_rdf"))

        # seed is the "kg"/target, which KgPipe.build will use when a task
        # declares an input named "kg".
        pipe = KgPipe(tasks=tasks, seed=target, data_dir=str(workdir), name=f"qap_mock_{cfg.family.value}")

        final = Data(path=workdir / "final.nt", format=final_format)
        return pipe, source, final

    if cfg.family == PipelineFamily.TEXT:
        text = Data(path=TEST_DATA_TEXT_PATH, format=DataFormat.TEXT)

        ie_impl = cfg.implementations.get("information_extraction", "pattern_ie")

        def _pattern_ie(inputs: dict[str, Data], outputs: dict[str, Data]) -> None:
            import re

            in_path = Path(inputs["input"].path)
            out_path = Path(outputs["output"].path)
            out_path.mkdir(parents=True, exist_ok=True)

            txt = _read_text(in_path)
            # Tiny, deterministic pattern extractor: "X is a Y" / "X is an Y".
            triples = []
            for m in re.finditer(r"([A-Z][A-Za-z0-9_ ]{2,40}) is an? ([A-Za-z][A-Za-z0-9_ -]{2,40})", txt):
                subj = m.group(1).strip()
                obj = m.group(2).strip()
                triples.append(
                    {
                        "subject": {"surface_form": subj},
                        "predicate": {"surface_form": "is_a"},
                        "object": {"surface_form": obj},
                    }
                )

            doc = {"text": txt[:10_000], "triples": triples, "chains": [], "links": []}
            (out_path / "pattern_ie.te.json").write_text(json.dumps(doc), encoding="utf-8")

        pattern_ie_task = KgTask(
            name="pattern_ie_extraction",
            input_spec={"input": DataFormat.TEXT},
            output_spec={"output": DataFormat.TE_JSON},
            function=_pattern_ie,
            description="Lightweight pattern IE (no docker)",
        )

        if ie_impl == "pattern_ie":
            tasks = [pattern_ie_task]
        else:
            use_docker = os.environ.get("QAP_MOCK_USE_DOCKER", "0") == "1"
            if use_docker:
                # Use CoreNLP OpenIE path for openie/hybrid/llm variants (docker-backed).
                tasks = [
                    Registry.get_task("corenlp_openie_extraction"),
                    Registry.get_task("corenlp_exchange"),
                ]
            else:
                # Default to the lightweight extractor when docker isn't enabled.
                tasks = [pattern_ie_task]

        pipe = KgPipe(tasks=tasks, seed=text, data_dir=str(workdir), name=f"qap_mock_{cfg.family.value}")
        # Many TE_JSON-producing tasks treat the output as a directory of documents.
        final = Data(path=workdir / "final_te", format=DataFormat.TE_JSON)
        return pipe, text, final

    raise ValueError(f"Unknown family: {cfg.family}")


def run_pipeline_for_config(
    *, cfg: PipelineConfig, base_workdir: Path, stable_files: bool = True
) -> PipelineRunResult:
    """
    Execute a real KGpipe pipeline for this config and return its artifacts.

    Results are cached by (family, cfg-hash) under base_workdir to avoid repeating
    expensive docker/service calls during search.
    """
    run_id = f"{cfg.family.value}_{_cfg_hash(cfg)}"
    workdir = base_workdir / run_id
    _ensure_dir(workdir)

    try:
        pipe, source, final = build_pipeline_for_config(cfg=cfg, workdir=workdir)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "KGpipe dependencies are not installed in this environment. "
            "To run the *real* (non-mock) execution path, install the project in editable mode:\n\n"
            "  python3 -m pip install -e .\n\n"
            "This will also install the `kgcore` dependency declared in `pyproject.toml`.\n"
            f"Original import error: {e}"
        ) from e

    # Apply env-var config mapping used by tasks.
    prev_env = _set_env_from_params(dict(cfg.params))
    try:
        # If final exists and stable_files=True, KgTask.run will skip; still ok.
        pipe.build(source=source, result=final, stable_files=stable_files)
        reports = pipe.run(stable_files_override=stable_files)
    finally:
        _restore_env(prev_env)

    return PipelineRunResult(
        family=cfg.family,
        cfg=cfg,
        workdir=workdir,
        final_output=final,
        task_reports=reports,
        aux={"source": str(source.path), "seed": str(pipe.seed.path), "run_id": run_id},
    )


def _read_text(path: Path, max_bytes: int = 4_000_000) -> str:
    # Keep it simple and avoid huge reads in case a docker task goes wild.
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def compute_rdf_metrics(*, output_nt: Path, ontology_ttl: Optional[Path] = None) -> dict[str, float]:
    import importlib

    try:
        rdflib = importlib.import_module("rdflib")
        Graph = getattr(rdflib, "Graph")
        URIRef = getattr(importlib.import_module("rdflib.term"), "URIRef")
        g = Graph()
        g.parse(output_nt, format="nt")
        triples = len(g)
    except Exception:
        # Fallback without rdflib: approximate triples by counting lines.
        txt = _read_text(output_nt)
        triples = len([ln for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")])
        Graph = None  # type: ignore[assignment]
        URIRef = None  # type: ignore[assignment]
        g = None  # type: ignore[assignment]

    # Consistency proxy: fraction of predicates that appear in ontology (or common RDF vocab).
    allowed: set[str] = set()
    if Graph is not None and URIRef is not None and ontology_ttl is not None and ontology_ttl.exists():
        try:
            og = Graph()
            og.parse(ontology_ttl)
            # Allow all predicates defined as properties + rdfs:label/rdf:type.
            for s, _, _ in og:
                # cheap heuristic: treat all subjects that are URIRefs as "allowed" predicates
                if isinstance(s, URIRef):
                    allowed.add(str(s))
            allowed.add("http://www.w3.org/2000/01/rdf-schema#label")
            allowed.add("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
        except Exception:
            allowed = set()

    if allowed and g is not None and URIRef is not None:
        ok = 0
        for _, p, _ in g:
            if isinstance(p, URIRef) and str(p) in allowed:
                ok += 1
        consistency = ok / max(1, triples)
    else:
        consistency = 0.5

    # Coverage proxy: normalize by union of input graphs when using bundled test data.
    try:
        src = Graph().parse(TEST_DATA_RDF_PATH, format="nt")
        tgt = Graph().parse(TEST_DATA_SEED_KG_PATH, format="nt")
        union_triples = len(src) + len(tgt)
        coverage = min(1.0, triples / max(1, union_triples))
    except Exception:
        coverage = min(1.0, triples / 10_000.0)

    # Accuracy proxy: reward non-trivial graphs (very small outputs are likely bad).
    accuracy = min(1.0, max(0.0, (triples / 2000.0)))

    return {"accuracy": float(accuracy), "coverage": float(coverage), "consistency": float(consistency)}


def compute_te_metrics(*, te_json_path: Path) -> dict[str, float]:
    """
    Compute lightweight metrics from TE_JSON outputs.

    This intentionally avoids requiring a gold standard. It's a pragmatic proxy:
    - coverage ~ extracted triples count
    - consistency ~ fraction of triples that have all 3 spans populated
    - accuracy ~ average link score if links exist, else a baseline
    """
    # TE_JSON may be a directory (many files) or a single file.
    triples = 0
    complete = 0
    link_scores: list[float] = []

    paths: Iterable[Path]
    if te_json_path.is_dir():
        paths = [p for p in te_json_path.iterdir() if p.is_file()]
    else:
        paths = [te_json_path]

    for p in paths:
        try:
            doc = json.loads(_read_text(p))
        except Exception:
            continue
        for t in doc.get("triples", []) or []:
            triples += 1
            s = (t.get("subject") or {}).get("surface_form")
            r = (t.get("predicate") or {}).get("surface_form")
            o = (t.get("object") or {}).get("surface_form")
            if s and r and o:
                complete += 1
        for l in doc.get("links", []) or []:
            try:
                link_scores.append(float(l.get("score", 0.0)))
            except Exception:
                pass

    # Normalize coverage against a rough scale for the bundled Hobbit text.
    coverage = min(1.0, triples / 5000.0)
    consistency = complete / max(1, triples) if triples else 0.0
    accuracy = (sum(link_scores) / len(link_scores)) if link_scores else 0.35
    accuracy = min(1.0, max(0.0, accuracy))

    return {"accuracy": float(accuracy), "coverage": float(coverage), "consistency": float(consistency)}


def default_base_workdir() -> Path:
    # Keep outputs inside the experiment folder by default.
    return Path(__file__).resolve().parents[2] / "output_qap_mock" / "_real_runs"


def make_temp_base_workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="qap_mock_real_"))


# - pipeline auto algo
# - cleaning
#   normalization task
# - pipeline task aggregation 
#   aggregate multiple task sub (DAGs) into a single task 
#   example: paris matching and fusion are two sub tasks, we can aggregate them into a single task

