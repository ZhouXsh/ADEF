import runpy
import sys
import types

from src.ADEF_pipeline_1pad_0721 import ADEFPipeline


def _install_compatibility_shim():
    pipeline_module = types.ModuleType("src.ADEF_pipeline_2pad_0721")
    pipeline_module.ADEFPipeline = ADEFPipeline
    sys.modules[pipeline_module.__name__] = pipeline_module


if __name__ == "__main__":
    _install_compatibility_shim()
    runpy.run_module("inference_2pad_0721", run_name="__main__")
