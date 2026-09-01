from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_protocol import (  # noqa: E402
    Sample, canonical_eat_filename, canonical_emotion, infer_emotion,
    parse_mead_meta,
)


def test_emotion_aliases():
    assert canonical_emotion("angry") == "anger"
    assert canonical_emotion("happy") == "happiness"
    assert canonical_emotion("disgusted") == "disgust"
    assert canonical_emotion("surprised") == "surprise"


def test_mead_path_and_eat_name():
    gt = "/data/MEAD/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4"
    assert infer_emotion(gt) == "anger"
    assert parse_mead_meta(gt) == ("M003", "ang", "3", "001")
    s = Sample(name="x", fake="/tmp/fake.mp4", gt=gt, emotion="anger")
    assert canonical_eat_filename(s, 0) == "0000_M003_ang_3_001.mp4"


def test_dfer_contempt_stays_semantic():
    gt = "/data/MEAD/videos/M003/front/contempt/level_3/M003_front_contempt_level_3_001.mp4"
    assert infer_emotion(gt) == "contempt"
