# coding: utf-8

"""
Global emotion category list used across the project.

Previously hard-coded in multiple files (e.g. train_emoClassifier.py,
train_emoEnhancer.py, ADEF_pipeline.py, ADEF_wrapper.py,
my_prepare/04_generate_template.py, run_alone.py). Centralized here so
any change to the supported emotions only needs to happen in one place.
"""

from typing import List


# The 8 emotion categories shared by the emotion classifier, enhancer,
# ADEF pipeline, template generation, and inference scripts.
global_emo_list: List[str] = [
    'angry',
    'contempt',
    'disgusted',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprised',
]
