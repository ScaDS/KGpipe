# kgcore/common/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

KGId = str

def is_uri(id: KGId) -> bool:
    # TODO improve later
    return id.startswith("http://") or id.startswith("https://")
"""
A unique internal identifier for any graph element
(entity, relation, or triple).
"""


@dataclass
class Lit:
    value: str
    datatype: Optional[str]


Props = Dict[str, Any | Lit]

def new_id() -> KGId:
    return str(uuid4())

@dataclass
class Result:
    """
    Lightweight return container for operations that may succeed or fail.

    Attributes:
        ok: True if the operation succeeded, False if it failed.
        value: The result value when ok is True; otherwise None.
        error: An error message or short description when ok is False.

    Notes:
        - Prefer returning Result over raising exceptions in low-level
          or backend code where control flow should not be interrupted.
        - Callers can check `if result.ok:` to branch on success.
    """
    ok: bool
    value: Any | None = None
    error: str | None = None
