"""
Python library parameter extraction from source code.
"""

import re
import ast
from typing import List, Optional, Union

from ..models import RawParameter, ExtractionResult, SourceType, ExtractionMethod
from ..base import RegexExtractor, LLMExtractor
from ..patterns import PYTHON_PATTERNS
from ..utils import normalize_parameter_name, parse_default_value


class PythonLibExtractor(RegexExtractor):
    """Extract parameters from Python code (functions, classes, docstrings)."""
    
    def __init__(self, use_llm: bool = False, llm_client=None):
        super().__init__(SourceType.PYTHON_LIB, PYTHON_PATTERNS)
        self.use_llm = use_llm
        if use_llm:
            self.llm_extractor = LLMPythonExtractor(llm_client)
    
    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """Extract parameters from Python source code."""
        if self.use_llm:
            return self.llm_extractor.extract(source, tool_name)
        
        parameters = []
        errors = []
        
        try:
            # Try to parse as Python AST
            try:
                tree = ast.parse(source)
                parameters.extend(self._extract_from_ast(tree, source))
            except SyntaxError:
                # If not valid Python, try regex-based extraction
                parameters.extend(self._extract_from_regex(source))
        
        except Exception as e:
            errors.append(f"Error extracting Python parameters: {str(e)}")
        
        return ExtractionResult(
            tool_name=tool_name or "unknown_python_lib",
            source_type=SourceType.PYTHON_LIB,
            extraction_method=ExtractionMethod.REGEX,
            parameters=parameters,
            errors=errors
        )
    
    # Type hints that almost certainly indicate I/O data, not configuration.
    _IO_TYPE_HINTS = frozenset({
        "DataFrame", "pd.DataFrame", "pandas.DataFrame",
        "ndarray", "np.ndarray", "numpy.ndarray",
        "Series", "pd.Series",
        "BaseTable", "BaseColumn",
        "Table", "Column",
        "Dataset",
        "Pool", "Process",
        "Iterator", "Generator",
        "TextIO", "BinaryIO", "IO",
    })

    @classmethod
    def _looks_like_io_param(cls, name: str, type_hint: Optional[str], has_default: bool) -> bool:
        """
        Heuristic: return True if a function parameter is likely an I/O
        argument rather than a tunable configuration knob.

        Rules:
        - Parameters whose type hint is a known data type (DataFrame, ndarray,
          BaseTable, etc.) are I/O.
        - Required parameters (no default) of non-__init__ methods whose names
          suggest data flow (source, target, input, output, data, table, path,
          pool, etc.) are I/O.
        """
        if type_hint:
            # Check the raw type and any component of a composite hint
            for io_type in cls._IO_TYPE_HINTS:
                if io_type in type_hint:
                    return True
        # Common I/O parameter name stems
        io_name_hints = {
            "source", "target", "input", "output", "data",
            "table", "column", "pool", "file", "path",
            "stream", "buffer", "reader", "writer",
        }
        normalized = name.lower().replace("_", "")
        for h in io_name_hints:
            if h in normalized:
                # If it has a simple scalar default, it might still be config
                if has_default:
                    return False
                return True
        return False

    def _extract_from_ast(self, tree: ast.AST, source: str) -> List[RawParameter]:
        """Extract parameters from Python AST."""
        parameters = []
        extractor_cls = self  # reference for nested class

        class ParameterVisitor(ast.NodeVisitor):
            def __init__(self):
                self.params = []
                self.source_lines = source.split('\n')
                self._current_class = None

            def visit_ClassDef(self, node):
                prev_class = self._current_class
                self._current_class = node.name

                # Extract class-level attributes (dataclasses, Pydantic models, etc.)
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        # Annotated assignment: name: type = default
                        if isinstance(item.target, ast.Name):
                            attr_name = item.target.id

                            # Get type hint
                            type_hint = None
                            if item.annotation:
                                type_hint = ast.unparse(item.annotation) if hasattr(ast, 'unparse') else str(item.annotation)

                            # Get default value
                            default_val = None
                            if item.value:
                                if hasattr(ast, 'unparse'):
                                    default_val = ast.unparse(item.value)
                                else:
                                    try:
                                        default_val = ast.literal_eval(item.value)
                                    except (ValueError, TypeError):
                                        default_val = None

                            param = RawParameter(
                                name=normalize_parameter_name(attr_name),
                                native_keys=[attr_name],
                                description=None,
                                type_hint=type_hint,
                                default_value=parse_default_value(str(default_val)) if default_val is not None else None,
                                required=default_val is None,
                                source=f"{node.name}.{attr_name}",
                                provenance={"class": node.name, "line": item.lineno if hasattr(item, 'lineno') else node.lineno}
                            )
                            self.params.append(param)
                    elif isinstance(item, ast.Assign):
                        # Regular assignment: name = value (might be in dataclass)
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                attr_name = target.id
                                # Try to get value
                                default_val = None
                                if item.value:
                                    try:
                                        default_val = ast.literal_eval(item.value)
                                    except (ValueError, TypeError):
                                        default_val = None

                                param = RawParameter(
                                    name=normalize_parameter_name(attr_name),
                                    native_keys=[attr_name],
                                    description=None,
                                    type_hint=None,
                                    default_value=parse_default_value(str(default_val)) if default_val is not None else None,
                                    required=False,
                                    source=f"{node.name}.{attr_name}",
                                    provenance={"class": node.name, "line": item.lineno}
                                )
                                self.params.append(param)

                self.generic_visit(node)
                self._current_class = prev_class

            def visit_FunctionDef(self, node):
                is_init = node.name == '__init__'
                is_method = self._current_class is not None
                class_name = self._current_class
                # For non-__init__ methods inside a class, only keep params
                # that look like configuration (have defaults and don't look
                # like I/O data arguments).
                skip_io = is_method and not is_init

                for arg in node.args.args:
                    if arg.arg in ('self', 'cls'):
                        continue

                    # Get type hint
                    type_hint = None
                    if arg.annotation:
                        type_hint = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)

                    # Get default value
                    default_val = None
                    default_idx = len(node.args.args) - len(node.args.defaults)
                    if arg in node.args.args[default_idx:]:
                        default_node = node.args.defaults[node.args.args[default_idx:].index(arg)]
                        if hasattr(ast, 'unparse'):
                            default_val = ast.unparse(default_node)
                        else:
                            default_val = ast.literal_eval(default_node) if isinstance(default_node, (ast.Constant, ast.Str, ast.Num)) else None

                    has_default = default_val is not None

                    # ── I/O filter for non-constructor methods ──
                    if skip_io and extractor_cls._looks_like_io_param(arg.arg, type_hint, has_default):
                        continue

                    # For non-__init__ methods, skip required params that
                    # have no default — they're almost always data args.
                    if skip_io and not has_default:
                        continue

                    # Extract docstring info (Sphinx :param: and numpydoc styles)
                    description = None
                    if ast.get_docstring(node):
                        docstring = ast.get_docstring(node)
                        # Sphinx style — :param name: description
                        sphinx_pat = re.compile(
                            rf":param\s+{re.escape(arg.arg)}:\s*(.+?)(?=\n|:param|$)",
                            re.MULTILINE,
                        )
                        m = sphinx_pat.search(docstring)
                        if m:
                            description = m.group(1).strip()
                        else:
                            # Numpydoc style —
                            #   name : type
                            #       Description text
                            numpydoc_pat = re.compile(
                                rf"^\s*{re.escape(arg.arg)}\s*(?::.*)?$\n((?:[ \t]+.+\n?)+)",
                                re.MULTILINE,
                            )
                            m = numpydoc_pat.search(docstring)
                            if m:
                                # Merge continuation lines and strip indent
                                desc_lines = [l.strip() for l in m.group(1).splitlines() if l.strip()]
                                description = " ".join(desc_lines)

                    func_label = f"{class_name}.{node.name}" if class_name else node.name
                    param = RawParameter(
                        name=normalize_parameter_name(arg.arg),
                        native_keys=[arg.arg],
                        description=description,
                        type_hint=type_hint,
                        default_value=parse_default_value(default_val) if default_val else None,
                        required=default_val is None,
                        source=f"{func_label}()",
                        provenance={
                            "function": node.name,
                            "class": class_name,
                            "is_constructor": is_init,
                            "line": node.lineno,
                        }
                    )
                    self.params.append(param)

                self.generic_visit(node)

        visitor = ParameterVisitor()
        visitor.visit(tree)
        return visitor.params
    
    def _extract_from_regex(self, source: str) -> List[RawParameter]:
        """Fallback regex-based extraction."""
        parameters = []
        
        # Extract function parameters
        func_pattern = re.compile(r"def\s+\w+\s*\(([^)]+)\)", re.MULTILINE)
        for match in func_pattern.finditer(source):
            params_str = match.group(1)
            for param_match in PYTHON_PATTERNS["function_param"].finditer(params_str):
                param_name = param_match.group(1)
                type_hint = param_match.group(2).strip() if param_match.group(2) else None
                default_val = param_match.group(3).strip() if param_match.group(3) else None
                
                parameters.append(RawParameter(
                    name=normalize_parameter_name(param_name),
                    native_keys=[param_name],
                    type_hint=type_hint,
                    default_value=parse_default_value(default_val) if default_val else None,
                    required=default_val is None,
                    source=match.group(0),
                    provenance={"method": "regex"}
                ))
        
        return parameters


class LLMPythonExtractor(LLMExtractor):
    """LLM-based Python parameter extraction."""
    
    def __init__(self, llm_client=None):
        super().__init__(SourceType.PYTHON_LIB, llm_client)
    
    def _create_prompt(self, source: str, tool_name: Optional[str] = None) -> str:
        return f"""Extract all configuration parameters from the following Python code.
Look for:
- Function parameters with type hints and defaults
- Class attributes with type annotations
- Configuration classes (dataclasses, Pydantic models)
- Environment variables

Python Code:
{source}

Return a JSON object with a 'parameters' array. Each parameter should have: name, native_keys, description, type_hint, default_value, required."""
    
    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """Extract parameters using LLM."""
        from pydantic import BaseModel
        from typing import List as TypingList
        
        class ParameterSchema(BaseModel):
            name: str
            native_keys: TypingList[str]
            description: Optional[str] = None
            type_hint: Optional[str] = None
            default_value: Optional[Union[str, int, float, bool]] = None
            required: bool = False
        
        class ExtractionSchema(BaseModel):
            parameters: TypingList[ParameterSchema]
        
        try:
            prompt = self._create_prompt(source, tool_name)
            response = self.llm_client.send_prompt(prompt, ExtractionSchema)
            
            parameters = []
            if "parameters" in response:
                for param_data in response["parameters"]:
                    raw_param = RawParameter(
                        name=normalize_parameter_name(param_data["name"]),
                        native_keys=param_data.get("native_keys", []),
                        description=param_data.get("description"),
                        type_hint=param_data.get("type_hint"),
                        default_value=param_data.get("default_value"),
                        required=param_data.get("required", False),
                        source=source[:200],  # First 200 chars
                        provenance={"method": "llm"}
                    )
                    parameters.append(raw_param)
            
            return ExtractionResult(
                tool_name=tool_name or "unknown_python_lib",
                source_type=SourceType.PYTHON_LIB,
                extraction_method=ExtractionMethod.LLM,
                parameters=parameters
            )
        except Exception as e:
            return ExtractionResult(
                tool_name=tool_name or "unknown_python_lib",
                source_type=SourceType.PYTHON_LIB,
                extraction_method=ExtractionMethod.LLM,
                parameters=[],
                errors=[f"LLM extraction failed: {str(e)}"]
            )


