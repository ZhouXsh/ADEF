import runpy
import sys
import types

from src.dataset.dataset_EmotionLevel_1pad_0721 import EmoLevelDataset
from src.modules.emotion_dit_Unification_1pad_0721 import DitTalkingHead


def _install_compatibility_shims():
    dataset_module = types.ModuleType(
        "src.dataset.dataset_EmotionLevel_2pad_0721"
    )
    dataset_module.EmoLevelDataset = EmoLevelDataset
    sys.modules[dataset_module.__name__] = dataset_module

    model_module = types.ModuleType(
        "src.modules.emotion_dit_Unification_2pad_0721"
    )
    model_module.DitTalkingHead = DitTalkingHead
    sys.modules[model_module.__name__] = model_module


if __name__ == "__main__":
    _install_compatibility_shims()
    if "--exp_name" not in sys.argv:
        sys.argv.extend(
            ["--exp_name", "20260721_emotion_dit_Unification_1pad_0721"]
        )
    runpy.run_module("train_2pad_0721", run_name="__main__")
