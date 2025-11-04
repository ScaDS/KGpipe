# kgcore/decorators/class_.py
from __future__ import annotations
from functools import wraps
from kgcore.system.recorder import SystemRecorder

_rec = SystemRecorder()

def class_(label: str, description: str = ""):
    def deco(cls):
        _rec.register_function(f"{cls.__module__}.{cls.__name__}", description or label)
        return cls
    return deco
