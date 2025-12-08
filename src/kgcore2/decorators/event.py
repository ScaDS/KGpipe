# kgcore/decorators/event.py
from __future__ import annotations
from functools import wraps
from datetime import datetime
import hashlib
import inspect
import pickle
from kgcore.system.recorder import SystemRecorder

_rec = SystemRecorder()

def _compute_function_hash(func):
    """Compute a hash of the function for change detection (inspired by joblib)."""
    try:
        # Get function source code
        source = inspect.getsource(func)
        
        # Get function bytecode
        code = func.__code__
        
        # Create a hash of the function's essential parts
        hash_input = f"{source}:{code.co_filename}:{code.co_firstlineno}:{code.co_argcount}:{code.co_kwonlyargcount}"
        
        # Include default values if any
        if func.__defaults__:
            hash_input += f":{func.__defaults__}"
        if func.__kwdefaults__:
            hash_input += f":{func.__kwdefaults__}"
        
        # Compute SHA-256 hash
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:16]
    except Exception:
        # Fallback to timestamp if hashing fails
        return datetime.utcnow().strftime("%Y%m%d%H%M%S")

def _get_timestamp_version():
    """Get a timestamp-based version string."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def event(name: str, description: str = "", version: str | None = None, 
          auto_version: str | None = None, **meta):
    """
    Decorator for tracking function execution in the system graph.
    
    Args:
        name: Human-readable name for the event
        description: Description of what the function does
        version: Explicit version string (overrides auto_version)
        auto_version: Automatic versioning mode:
            - "timestamp": Use current timestamp as version
            - "hash": Use function hash as version
            - None: No automatic versioning
        **meta: Additional metadata to store with the event
    """
    def deco(fn):
        # Determine version
        if version is not None:
            final_version = version
        elif auto_version == "timestamp":
            final_version = _get_timestamp_version()
        elif auto_version == "hash":
            final_version = _compute_function_hash(fn)
        else:
            final_version = None
        
        # Add version to metadata if specified
        event_meta = meta.copy()
        if final_version is not None:
            event_meta["version"] = final_version
        
        fn_id = _rec.register_function(f"{fn.__module__}.{fn.__qualname__}", description or name).id
        
        @wraps(fn)
        def inner(*args, **kwargs):
            run_id = _rec.start_run(fn_id, {"name": name, **event_meta})
            try:
                res = fn(*args, **kwargs)
                _rec.finish_run(run_id, {"result": "ok"})
                return res
            except Exception as e:
                _rec.finish_run(run_id, {"result": "error", "error": str(e)})
                raise
        return inner
    return deco
