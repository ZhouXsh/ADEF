"""
使用方法（不同方法、相同情感对比图 / 左面板）：
1. 修改下面 METHODS 中每个方法对应的视频路径；这些视频应尽量使用同一 source、同一音频和同一目标情感。
2. 修改 OUTPUT_PATH、NUM_SAMPLES 等参数；默认 LABEL_AXIS="x"，严格对应 visual/readme.md：横轴是方法名，纵轴是采样位置。
3. 在仓库根目录执行：python visual/compare_methods_grid.py
4. 输出为无损 PNG。若想排成你给出的参考图那样“方法在纵轴、时间在横轴”，只需把 LABEL_AXIS 改为 "y"。

说明：脚本会先取所有输入视频的共同有效时长，再在同一组绝对时间戳上间隔采样，因此不同 FPS 的视频也能尽量做到同一时刻对齐。
"""

from collections import OrderedDict

from grid_utils import build_video_grid

base_path = '/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake'
video_name = 'M003_front_angry_level_3_015_M003_front_angry_level_3_015_angry.mp4'

# ========================= 只需要修改这里 =========================
METHODS = OrderedDict(
    [
        ("GT", "/home/Zhouxishi/VirtualMan_proj/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_015.mp4"),
        ("Wav2Lip", f"{base_path}/wav2lip/{video_name}"),
        ("KDTalker", f"{base_path}/kdtalker/{video_name}"),
        ("FlashHead", f"{base_path}/SoulX-FlashHead-Pro/{video_name}"),
        ("EAT", f"{base_path}/eat_code/{video_name}"),
        ("DICE-Talk", f"{base_path}/DICE_TALK_MEAD_FULL/{video_name}"),
        ("Fantasy-Talking", f"{base_path}/Fantasy_Subset/{video_name}"),
        ("HSA-Motion (Ours)", f"{base_path}/20260909_fusion_balanced_decay_ema_20cfg_full/{video_name}"),
    ]
)

OUTPUT_PATH = "visual/results/method_comparison.png"
PANEL_TITLE = "Same Emotion Comparison"
NUM_SAMPLES = 5
START_RATIO = 0.10
END_RATIO = 0.90
CELL_WIDTH = 256
CELL_HEIGHT = 256
LABEL_AXIS = "x"  # "x" 符合 readme；"y" 更接近你给出的参考图布局
FONT_PATH = None  # 中文标签可指定中文字体，例如 /usr/share/fonts/.../NotoSansCJK-Regular.ttc
# ================================================================


if __name__ == "__main__":
    output = build_video_grid(
        METHODS,
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
    print(f"Saved method-comparison grid to: {output}")
