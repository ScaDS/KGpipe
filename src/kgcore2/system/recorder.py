# kgcore/system/recorder.py
from __future__ import annotations
from datetime import datetime
from kgcore.api import KG
from kgcore.system.schema import SysLabels

class SystemRecorder:
    def __init__(self, kg=None):
        self.kg = kg or KG()

    def register_function(self, qualname: str, description: str | None = None):
        return self.kg.create_entity([SysLabels.CodeObject, SysLabels.Function],
                                     {"name": qualname, "description": description or ""})

    def start_run(self, func_id, meta=None):
        run = self.kg.create_entity([SysLabels.Run], {"ts": datetime.utcnow().isoformat(), **(meta or {})})
        self.kg.create_relation("CALLS", run.id, func_id, {})
        return run.id

    def finish_run(self, run_id, meta=None):
        self.kg.add_meta(run_id, {"status": "ok", **(meta or {})})
