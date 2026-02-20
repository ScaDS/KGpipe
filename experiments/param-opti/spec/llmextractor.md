# LLM Extractors

## Overview

Each source type (CLI, Python, Docker, HTTP API, README) has a **regex extractor** and an **LLM extractor**. The LLM variants exist as a fallback: when the regex-based approach finds few or no parameters, the `ParameterMiner` can automatically delegate to the LLM version if an `llm_client` is available.

## Class Hierarchy

```
BaseExtractor (ABC)                    # extraction/base.py
├── RegexExtractor                     # pattern-based, offline
│   ├── CLIExtractor
│   ├── PythonLibExtractor
│   ├── HTTPAPIExtractor
│   ├── DockerExtractor
│   └── ReadmeDocExtractor
└── LLMExtractor                       # LLM-based, requires kgpipe_llm
    ├── LLMCLIExtractor
    ├── LLMPythonExtractor
    ├── LLMHTTPExtractor
    ├── LLMDockerExtractor
    └── LLMReadmeDocExtractor
```

All extractors share the same interface:

```python
def extract(self, source: str, tool_name: Optional[str] = None) -> ExtractionResult
```

## How LLMExtractor Works

### 1. Initialization (`base.py`)

```python
class LLMExtractor(BaseExtractor):
    def __init__(self, source_type: SourceType, llm_client=None):
        self.llm_client = llm_client
        if llm_client is None:
            from kgpipe_llm.common.core import get_client_from_env
            self.llm_client = get_client_from_env()
```

If no `llm_client` is passed, the constructor tries to auto-create one via `get_client_from_env()`, which reads these environment variables:

| Variable               | Purpose                                 |
|------------------------|-----------------------------------------|
| `LLM_ENDPOINT_URL`     | API endpoint (Ollama or OpenAI-compat)  |
| `DEFAULT_LLM_MODEL_NAME` | Model name (`gemma3:27B`, `gpt-4o`, …)|
| `OLLAMA_TOKEN`         | Token for Ollama API                    |
| `OPENAI_TOKEN`         | Token for OpenAI API                    |
| `LLM_SEED`             | Optional reproducibility seed           |
| `CONTEXT_WINDOW`       | Max context window (default 16384)      |

The client auto-detects whether to use the **OpenAI** or **Ollama** backend based on the model name.

### 2. Prompt Construction (`_create_prompt`)

Each LLM extractor overrides `_create_prompt()` with a source-type-specific prompt template. For example, `LLMCLIExtractor`:

```
Extract all configuration parameters from the following CLI help output.
For each parameter, identify:
- Parameter name (normalized, without -- or -)
- Native keys/flags (--flag, -f, etc.)
- Description
- Type (if mentioned)
- Default value (if mentioned)
- Whether it's required or optional

CLI Help Output:
{source}

Return a JSON object with a 'parameters' array. Each parameter should have:
name, native_keys, description, type_hint, default_value, required.
```

Each source type adapts the prompt to mention the kind of content it expects (code blocks for README, ENV/ARG for Docker, function signatures for Python, etc.).

### 3. Structured Output via Pydantic Schema

The `extract()` method defines a Pydantic schema inline and passes it to `send_prompt()`:

```python
class ParameterSchema(BaseModel):
    name: str
    native_keys: List[str]
    description: Optional[str] = None
    type_hint: Optional[str] = None
    default_value: Optional[Union[str, int, float, bool]] = None
    required: bool = False

class ExtractionSchema(BaseModel):
    parameters: List[ParameterSchema]

response = self.llm_client.send_prompt(prompt, ExtractionSchema)
```

`LLMClient.send_prompt()` uses the Pydantic model's JSON schema to enforce structured output:
- **OpenAI backend**: uses tool/function calling (`openai_call_with_tool`) to get schema-conformant JSON.
- **Ollama backend**: passes the schema in the `format` field so the model outputs valid JSON.

The response is always a `dict` with a `"parameters"` key containing a list of parameter objects.

### 4. Response Parsing

The returned dict is iterated and each entry is converted to a `RawParameter`:

```python
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
```

The result is wrapped in an `ExtractionResult` with `extraction_method=ExtractionMethod.LLM`.

### 5. Error Handling

All LLM extractors catch exceptions and return an empty `ExtractionResult` with the error message in the `errors` list. This ensures a failed LLM call never crashes the extraction pipeline.

## How ParameterMiner Uses LLM Extractors

In `ParameterMiner.extract_parameters()`, the `method` argument controls dispatch:

| Method                  | Behavior                                              |
|-------------------------|-------------------------------------------------------|
| `ExtractionMethod.REGEX`| Always use the regex extractor.                       |
| `ExtractionMethod.LLM`  | Always use the LLM extractor (raises if no client).   |
| `ExtractionMethod.AUTO` | Try regex first. If it returns 0 parameters **and** an `llm_client` is available, fall back to the LLM extractor. |

The experiment runner (`run_experiment.py --use-llm`) sets `use_llm=True`, which provides an `llm_client` to the `ParameterMiner`, enabling the AUTO fallback.

## Existing LLM Extractors

| Class                    | Source Type | Prompt Focus                                  |
|--------------------------|-------------|-----------------------------------------------|
| `LLMCLIExtractor`        | CLI         | Flags, options, defaults from help output     |
| `LLMPythonExtractor`     | Python      | Function params, class attrs, env vars        |
| `LLMHTTPExtractor`       | HTTP API    | Query/path/body/header params from specs      |
| `LLMDockerExtractor`     | Docker      | ENV, ARG, volumes, ports                      |
| `LLMReadmeDocExtractor`  | README      | Flags, env vars, config keys, tunable values  |

## Testing

LLM extractors are tested with a mock client (`conftest.py::mock_llm_client`) that returns a canned response, so tests run without a live LLM endpoint.

