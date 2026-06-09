from kgcore.api.ontology import Ontology, OntologyUtil
from kgpipe.common import Data, DataFormat, Registry
from kgpipe_llm.common.snippets import generate_ontology_snippet_v3
from kgpipe_llm.common.core import get_client_from_env
from kgpipe_llm.json_mapping import JSON_Mapping_v1
import os
from pathlib import Path
from typing import Dict
from kgpipe_llm_examples.struct_out_schema import RDFTriples
from rdflib import Graph, URIRef, Literal
import json

ONTOLOGY = None
def get_ontology() -> Ontology:
    global ONTOLOGY
    if ONTOLOGY is None:
        ontology_path = os.getenv("ONTOLOGY_PATH")
        if ontology_path is None:
            raise ValueError("ONTOLOGY is not set")
        ONTOLOGY = OntologyUtil.load_ontology_from_file(Path(ontology_path))
    return ONTOLOGY

def generate_python_mapping_file(json_data: str, ontology_snippet: str) -> str:

    system_prompt = """
    You are a senior Python engineer and RDF/OWL practitioner. You write production-grade, testable, deterministic code. You never invent ontology terms or JSON fields not present in the provided inputs.
    """
    
    instructions=f"""

I need you to generate a single Python file named `converter.py` that converts JSON documents (all with the same structure) into RDF using the provided ontology snippet.

The script must be runnable exactly as:
    python converter.py INPUT.json OUTPUT.rdf

========================
INPUTS
========================

Example JSON (structure representative of all files):

```
{json_data}
```

Ontology snippet (Turtle or RDF/XML, may be partial):

```
{ontology_snippet}
```

========================
FUNCTIONAL REQUIREMENTS
========================

1) RDF Library
   - Use rdflib for RDF graph creation and serialization.
   - Only use Python standard library + rdflib. No other dependencies.

2) Output Format
   - If OUTPUT filename ends with:
        .ttl  -> serialize as Turtle
        .nt   -> serialize as N-Triples
        .rdf or .xml -> serialize as RDF/XML
   - Default to Turtle if extension is unknown.

3) Strict Ontology Compliance (NO HALLUCINATION)
   - Use ONLY classes and properties that explicitly exist in $ONTOLOGY.
   - Do NOT invent new ontology terms.
   - Do NOT assume ontology terms beyond what is provided.
   - If a JSON field has no clear mapping to a term in the ontology snippet:
        - Print a warning to stderr
        - Skip that field
   - Document skipped mappings in comments.

4) Explicit Mapping Section
   - Include a clearly structured MAPPING dictionary (or equivalent structure)
     that documents:
        - JSON field (or JSON path)
        - Corresponding ontology class/property URI
        - Whether it maps to:
            * rdf:type
            * literal
            * object property
            * repeated property (array)
   - If you must infer mapping decisions, explain them in code comments.
   - Keep mappings conservative and minimal.

5) URI Strategy (Deterministic)
   - If JSON contains an ID field, use it to build subject URIs.
   - If no ID field exists:
        - Generate deterministic URIs using a stable hash
        - Hash a canonicalized JSON subset (describe in comments what is hashed)
   - Do NOT generate random UUIDs.

6) Datatype Handling
   - Map Python types as:
        int      -> xsd:integer
        float    -> xsd:decimal
        bool     -> xsd:boolean
   - ISO-8601 date strings:
        -> xsd:date or xsd:dateTime if clearly valid
   - Otherwise use plain literals.
   - Import and bind xsd namespace properly.

7) Nested Structures
   - Arrays of primitives:
        -> repeat the mapped predicate with multiple literals
   - Arrays of objects:
        -> create separate resources with URIs
        -> link them using object properties
   - Nested objects:
        -> create new resources and link appropriately

8) Graph Construction
   - Use rdf:type for class membership.
   - If ontology contains rdfs:label and JSON has a name/title-like field,
     map it appropriately.
   - Bind all prefixes found in the ontology snippet where possible.
   - Also bind rdf, rdfs, xsd, owl namespaces.

9) Input Validation
   - Accept either:
        a) A single JSON object at root
        b) A list of JSON objects at root
   - If required top-level keys from the example JSON are missing,
     raise a clear error and exit with non-zero status.
   - If ontology parsing fails, print a helpful error and exit non-zero.
   - Exit code 0 on success.

10) CLI Behavior
   - Use argparse for argument parsing.
   - Required positional arguments:
        INPUT
        OUTPUT
   - Print helpful usage message.

11) Code Quality
   - Include:
        - Module docstring explaining purpose
        - Clear function separation:
            * load_json(...)
            * load_ontology(...)
            * build_graph(...)
            * serialize_graph(...)
            * main()
        - Docstrings for major functions
        - Inline comments explaining key logic
   - Keep file self-contained.
   - Make code easy to extend.

========================
DELIVERABLE FORMAT
========================

Output ONLY the complete contents of converter.py.
Return it as a single Python code block.
Do NOT include explanations, commentary, or text outside the code block.
"""

    llm = get_client_from_env()
    response = llm.send_prompt(instructions, "", system_prompt)
    print("-GENERATED CODE--------------------------------")
    print(response)
    print("-GENERATED CODE--------------------------------")
    return str(response)


import os
import re
import sys
import json
import shlex
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List


FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*\n(?P<code>.*?\n)```",
    re.DOTALL,
)

SHEBANG_RE = re.compile(r"^#!.*python[0-9.]*\s*$", re.IGNORECASE)


def extract_python_code(llm_text: str) -> str:
    """
    Extract Python code from an LLM response that may include Markdown code fences
    and extra commentary.

    Strategy:
    1) Prefer fenced blocks labeled python/py.
    2) Else, take the largest fenced block (any language).
    3) Else, fallback: try to strip obvious prose and keep lines that look like code.
    """
    text = llm_text.strip()

    matches = list(FENCE_RE.finditer(text))
    if matches:
        # Prefer python-labeled blocks
        py_blocks: List[str] = []
        other_blocks: List[str] = []

        for m in matches:
            lang = (m.group("lang") or "").strip().lower()
            code = m.group("code")
            # Remove trailing newline added by regex capture style
            code = code.rstrip("\n")
            if lang in ("python", "py"):
                py_blocks.append(code)
            else:
                other_blocks.append(code)

        if py_blocks:
            # If multiple python blocks, join with spacing (common if imports + code split)
            return "\n\n".join(b.strip() for b in py_blocks).strip()

        # Otherwise, choose the largest block (often the actual file)
        largest = max((m.group("code") for m in matches), key=lambda s: len(s))
        return largest.strip()

    # No fenced code blocks — heuristic fallback.
    # Keep lines that look like code; drop obvious narrative lines.
    lines = text.splitlines()

    kept: List[str] = []
    for line in lines:
        s = line.rstrip()

        # Drop empty leading/trailing prose later; keep empties inside code
        # Drop markdown headings / bullets likely to be prose
        if re.match(r"^\s*(#+\s+|[-*]\s+|\d+\)\s+)", s):
            continue

        # Keep shebang, imports, defs, classes, assignments, indented blocks, etc.
        looks_like_code = (
            bool(SHEBANG_RE.match(s))
            or bool(re.match(r"^\s*(from\s+\S+\s+import\s+|import\s+\S+)", s))
            or bool(re.match(r"^\s*(def|class)\s+\w+\s*\(", s))
            or bool(re.match(r"^\s*(if|elif|else|for|while|try|except|with)\b", s))
            or bool(re.match(r"^\s*@\w+", s))  # decorators
            or bool(re.match(r"^\s*\w+\s*=", s))
            or s.startswith("    ") or s.startswith("\t")  # indented code
        )
        if looks_like_code:
            kept.append(s)
        else:
            # If line contains common code tokens, keep it (last resort)
            if any(tok in s for tok in ("argparse", "rdflib", "Graph(", "def ", "__name__")):
                kept.append(s)

    code = "\n".join(kept).strip()
    if not code:
        raise ValueError("Could not extract Python code from LLM response.")
    return code


def normalize_code_for_file(code: str) -> str:
    """
    Minor cleanup: remove stray triple backticks if they survived,
    and strip leading/trailing whitespace.
    """
    code = code.replace("```python", "").replace("```py", "").replace("```", "")
    return code.strip() + "\n"


def write_python_file(code: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")


def run_converter(
    converter_path: Path,
    input_json: Path,
    output_rdf: Path,
    python_exe: Optional[str] = None,
    cwd: Optional[Path] = None,
    timeout_sec: int = 120,
) -> subprocess.CompletedProcess:
    """
    Execute: python converter.py input.json output.rdf
    Captures stdout/stderr for inspection.
    """
    python_exe = python_exe or sys.executable
    cmd = [python_exe, str(converter_path), str(input_json), str(output_rdf)]

    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def build_and_run_converter(
    llm_response: str,
    input_json_path: str,
    output_rdf_path: str,
    out_converter_path: Optional[str] = None,
) -> Tuple[Path, subprocess.CompletedProcess]:
    """
    End-to-end:
    - Extract code from LLM response
    - Write converter.py (either to out_converter_path, or a temp dir)
    - Run it
    - Return (path_to_converter, completed_process)
    """
    code = extract_python_code(llm_response)
    code = normalize_code_for_file(code)

    input_json = Path(input_json_path).resolve()
    output_rdf = Path(output_rdf_path).resolve()

    if out_converter_path:
        converter_path = Path(out_converter_path).resolve()
        write_python_file(code, converter_path)
        proc = run_converter(converter_path, input_json, output_rdf)
        return converter_path, proc

    with tempfile.TemporaryDirectory(prefix="llm_converter_") as td:
        td_path = Path(td)
        converter_path = td_path / "converter.py"
        write_python_file(code, converter_path)

        proc = run_converter(converter_path, input_json, output_rdf, cwd=td_path)

        # If you want to persist the converter for debugging, copy it out here.
        # For now we return a path that will no longer exist after context exit,
        # so if you need persistence, pass out_converter_path instead.
        return converter_path, proc

@Registry.task(input_spec={"input": DataFormat.JSON}, output_spec={"output": DataFormat.RDF_NTRIPLES})
def mapping_python_target(inputs: Dict[str, Data], outputs: Dict[str, Data]) -> None:

    json_data = json.dumps(json.load(open(inputs["input"].path)), indent=4)
    ontology_snippet = generate_ontology_snippet_v3(get_ontology())

    python_mapping_file_response = generate_python_mapping_file(json_data, ontology_snippet)
    
    converter_path, proc = build_and_run_converter(python_mapping_file_response, inputs["input"].path, outputs["output"].path)