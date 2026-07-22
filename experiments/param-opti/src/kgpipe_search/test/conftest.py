import sys
import types
from importlib import import_module


def _install_param_opti_shim() -> None:
    if "param_opti" in sys.modules:
        return

    param_opti = types.ModuleType("param_opti")
    tasks = types.ModuleType("param_opti.tasks")

    for lib in (
        "base_linker_lib",
        "base_matcher_lib",
        "paris_lib",
        "fusion_lib",
        "spotlight_lib",
        "corenlp_lip",
        "genie_lib",
    ):
        module = import_module(f"kgpipe_search.dev.tasks.{lib}")
        setattr(tasks, lib, module)
        sys.modules[f"param_opti.tasks.{lib}"] = module

    param_opti.tasks = tasks
    sys.modules["param_opti"] = param_opti
    sys.modules["param_opti.tasks"] = tasks


_install_param_opti_shim()
