"""
README / documentation parameter extraction.

Extracts configuration parameters from README files, documentation pages,
and other unstructured markdown/text docs that describe tool usage.
"""

import re
from typing import List, Optional, Set, Union

from ..models import RawParameter, ExtractionResult, SourceType, ExtractionMethod
from ..base import RegexExtractor, LLMExtractor
from ..patterns import README_PATTERNS
from ..utils import normalize_parameter_name, parse_default_value


# Noise words that appear as flags/placeholders but are not real parameters
_NOISE_NAMES: Set[str] = {
    "h", "help", "version", "v", "verbose", "quiet", "q",
    "the", "a", "an", "is", "are", "was", "were", "be",
    "to", "of", "in", "for", "on", "at", "by", "with",
    "it", "its", "we", "our", "you", "your",
    "e", "g", "i", "x", "s",
}


def _extract_code_blocks(text: str) -> List[str]:
    """Return contents of fenced code blocks (``` … ```)."""
    return re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)


class ReadmeDocExtractor(RegexExtractor):
    """Extract parameters from README / documentation text (Markdown or plain text)."""

    def __init__(self, use_llm: bool = False, llm_client=None):
        super().__init__(SourceType.README, README_PATTERNS)
        self.use_llm = use_llm
        if use_llm:
            self.llm_extractor = LLMReadmeDocExtractor(llm_client)

    def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult:
        """Extract parameters from README / documentation text."""
        if self.use_llm:
            return self.llm_extractor.extract(source, tool_name)

        parameters: List[RawParameter] = []
        errors: List[str] = []
        seen_names: Set[str] = set()

        try:
            # --- 1. Parameters described in markdown list items ---
            # e.g.  - `threshold`: The matching threshold (default: 0.5)
            for match in README_PATTERNS["list_param"].finditer(source):
                name_raw = match.group(1)
                description = match.group(2).strip()
                normalized = normalize_parameter_name(name_raw)
                if normalized in _NOISE_NAMES or len(normalized) < 2:
                    continue
                if normalized in seen_names:
                    continue
                seen_names.add(normalized)

                default_val = self._find_default(description)
                type_hint = self._find_type_hint(description)

                parameters.append(RawParameter(
                    name=normalized,
                    native_keys=[name_raw],
                    description=description,
                    type_hint=type_hint,
                    default_value=parse_default_value(default_val) if default_val else None,
                    required=False,
                    source=match.group(0).strip(),
                    provenance={"method": "readme_list_param"},
                ))

            # --- 2. Parameters in markdown tables ---
            for match in README_PATTERNS["table_param"].finditer(source):
                name_raw = match.group(1).strip()
                col2 = match.group(2).strip()
                col3 = match.group(3).strip()
                normalized = normalize_parameter_name(name_raw)
                if normalized in _NOISE_NAMES or len(normalized) < 2:
                    continue
                if normalized in seen_names:
                    continue
                seen_names.add(normalized)

                # Heuristic: second column is often type, third is description
                type_hint = col2 if col2 and len(col2) < 30 else None
                description = col3 or col2

                parameters.append(RawParameter(
                    name=normalized,
                    native_keys=[name_raw],
                    description=description if description else None,
                    type_hint=type_hint,
                    default_value=None,
                    required=False,
                    source=match.group(0).strip(),
                    provenance={"method": "readme_table"},
                ))

            # --- 3. Flags from code blocks ---
            code_blocks = _extract_code_blocks(source)
            for block in code_blocks:
                for match in README_PATTERNS["code_block_flag"].finditer(block):
                    flag = match.group(1)
                    value = match.group(2)
                    normalized = normalize_parameter_name(flag)
                    if normalized in _NOISE_NAMES or len(normalized) < 2:
                        continue
                    if normalized in seen_names:
                        continue
                    seen_names.add(normalized)

                    parameters.append(RawParameter(
                        name=normalized,
                        native_keys=[flag],
                        description=None,
                        type_hint=None,
                        default_value=parse_default_value(value) if value else None,
                        required=False,
                        source=block[:200].strip(),
                        provenance={"method": "readme_code_block"},
                    ))

                # JVM-style flags
                for jvm_match in README_PATTERNS["jvm_flag"].finditer(block):
                    flag = jvm_match.group(1)
                    normalized = normalize_parameter_name(flag)
                    if normalized in seen_names:
                        continue
                    seen_names.add(normalized)

                    parameters.append(RawParameter(
                        name=normalized,
                        native_keys=[flag],
                        description=f"JVM flag: {flag}",
                        type_hint=None,
                        default_value=None,
                        required=False,
                        source=block[:200].strip(),
                        provenance={"method": "readme_jvm_flag"},
                    ))

            # --- 4. Inline flags referenced with backticks ---
            for match in README_PATTERNS["inline_flag"].finditer(source):
                flag = match.group(1)
                normalized = normalize_parameter_name(flag)
                if normalized in _NOISE_NAMES or len(normalized) < 2:
                    continue
                if normalized in seen_names:
                    continue
                seen_names.add(normalized)

                # Try to find a surrounding sentence as description
                start = max(0, match.start() - 120)
                end = min(len(source), match.end() + 120)
                context = source[start:end].replace("\n", " ").strip()

                parameters.append(RawParameter(
                    name=normalized,
                    native_keys=[flag],
                    description=context,
                    type_hint=None,
                    default_value=None,
                    required=False,
                    source=context,
                    provenance={"method": "readme_inline_flag"},
                ))

            # --- 5. Environment variable references ---
            for match in README_PATTERNS["env_reference"].finditer(source):
                var_name = match.group(1)
                var_value = match.group(2) if match.lastindex >= 2 else None
                normalized = normalize_parameter_name(var_name)
                if normalized in seen_names or len(normalized) < 2:
                    continue
                seen_names.add(normalized)

                parameters.append(RawParameter(
                    name=normalized,
                    native_keys=[var_name],
                    description=f"Environment variable: {var_name}",
                    type_hint=None,
                    default_value=parse_default_value(var_value) if var_value else None,
                    required=False,
                    source=match.group(0).strip(),
                    provenance={"method": "readme_env_var"},
                ))

            # --- 6. Placeholder parameters from usage lines ---
            for match in README_PATTERNS["placeholder"].finditer(source):
                name_raw = match.group(1)
                normalized = normalize_parameter_name(name_raw)
                if normalized in _NOISE_NAMES or len(normalized) < 2:
                    continue
                if normalized in seen_names:
                    continue
                seen_names.add(normalized)

                # Grab surrounding line as context
                line_start = source.rfind("\n", 0, match.start()) + 1
                line_end = source.find("\n", match.end())
                if line_end == -1:
                    line_end = len(source)
                context_line = source[line_start:line_end].strip()

                parameters.append(RawParameter(
                    name=normalized,
                    native_keys=[f"<{name_raw}>"],
                    description=context_line,
                    type_hint=None,
                    default_value=None,
                    required=True,  # placeholders are usually required
                    source=context_line,
                    provenance={"method": "readme_placeholder"},
                ))

        except Exception as e:
            errors.append(f"Error extracting README parameters: {str(e)}")

        return ExtractionResult(
            tool_name=tool_name or "unknown_readme",
            source_type=SourceType.README,
            extraction_method=ExtractionMethod.REGEX,
            parameters=parameters,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_default(text: str) -> Optional[str]:
        """Try to extract a default value from a description string."""
        m = re.search(r"default[=:]\s*[`\"']?([^`\"'\]),\s]+)", text, re.IGNORECASE)
        return m.group(1) if m else None

    @staticmethod
    def _find_type_hint(text: str) -> Optional[str]:
        """Try to infer a type hint from a description string."""
        for token in ("int", "integer", "float", "number", "bool", "boolean", "string", "str", "path", "file"):
            if re.search(rf"\b{token}\b", text, re.IGNORECASE):
                return token
        return None


class LLMReadmeDocExtractor(LLMExtractor):
    """LLM-based README / documentation parameter extraction."""

    def __init__(self, llm_client=None):
        super().__init__(SourceType.README, llm_client)

    def _create_prompt(self, source: str, tool_name: Optional[str] = None) -> str:
        return f"""Extract all configuration parameters from the following README / documentation text.
Look for:
- Command-line flags and options mentioned in usage examples
- Environment variables
- Configuration keys or settings
- Input/output paths that can be parameterized
- Any tunable values (thresholds, limits, memory sizes, etc.)

Documentation:
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
                        source=source[:200],
                        provenance={"method": "llm"},
                    )
                    parameters.append(raw_param)

            return ExtractionResult(
                tool_name=tool_name or "unknown_readme",
                source_type=SourceType.README,
                extraction_method=ExtractionMethod.LLM,
                parameters=parameters,
            )
        except Exception as e:
            return ExtractionResult(
                tool_name=tool_name or "unknown_readme",
                source_type=SourceType.README,
                extraction_method=ExtractionMethod.LLM,
                parameters=[],
                errors=[f"LLM extraction failed: {str(e)}"],
            )

