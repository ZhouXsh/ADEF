"""
使用方法（ADEF 不同情感类型对比图 / 右面板）：
1. 修改下面 EMOTIONS 中每个情感对应的视频路径；这些视频建议使用同一 source、同一音频，只改变情感条件。
2. 修改 OUTPUT_PATH、NUM_SAMPLES 等参数；默认 LABEL_AXIS="x"，严格对应 visual/readme.md：横轴是情感类型，纵轴是采样位置。
3. 在仓库根目录执行：python visual/compare_emotions_grid.py
4. 输出为无损 PNG。若想排成你给出的参考图那样“情感在纵轴、时间在横轴”，只需把 LABEL_AXIS 改为 "y"。

说明：脚本会在所有情感视频的共同有效时长上使用同一组绝对时间戳进行采样，方便直接比较情感强度和面部动态差异。
"""

from collections import OrderedDict

from grid_utils import build_video_grid


# ========================= 只需要修改这里 =========================
EMOTIONS = OrderedDict(
    [
        ("Neutral", "/path/to/neutral.mp4"),
        ("Happy", "/path/to/happy.mp4"),
        ("Angry", "/path/to/angry.mp4"),
        ("Surprised", "/path/to/surprised.mp4"),
        ("Sad", "/path/to/sad.mp4"),
    ]
)

OUTPUT_PATH = "visual/results/emotion_comparison.png"
PANEL_TITLE = "ADEF Emotion Control"
NUM_SAMPLES = 5
START_RATIO = 0.10
END_RATIO = 0.90
CELL_WIDTH = 256
CELL_HEIGHT = 256
LABEL_AXIS = "x"  # "x" 符合 readme；"y" 更接近你给出的参考图布局
FONT_PATH = None
# ================================================================


if __name__ == "__main__":
    output = build_video_grid(
        EMOTIONS,
        OUTPUT_PATH,
        num_samples=NUM_SAMPLES,
        start_ratio=START_RATIO,
        end_ratio=END_RATIO,
        cell_width=CELL_WIDTH,
        cell_height=CELL_HEIGHT,
        label_axis=LABEL_AXIS,
        font_path=FONT_PATH,
        panel_title=PANEL_TITLE,
    )
    print(f"Saved emotion-comparison grid to: {output}")
