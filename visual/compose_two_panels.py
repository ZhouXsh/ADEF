"""
使用方法（把两张网格图拼成论文中的左右双面板）：
1. 先执行 compare_methods_grid.py，生成左侧“不同方法、相同情感”对比图。
2. 再执行 compare_emotions_grid.py，生成右侧“ADEF 不同情感类型”对比图。
3. 根据需要修改 LEFT_PANEL、RIGHT_PANEL 和 OUTPUT_PATH。
4. 在仓库根目录执行：python visual/compose_two_panels.py

脚本只负责无损拼接，不会再次缩放内部网格；较矮的一侧会在白色背景上垂直居中。
"""

from pathlib import Path

from PIL import Image


# ========================= 只需要修改这里 =========================
LEFT_PANEL = "visual/results/method_comparison.png"
RIGHT_PANEL = "visual/results/emotion_comparison.png"
OUTPUT_PATH = "visual/results/combined_comparison.png"
GAP = 24
OUTER_PADDING = 8
# ================================================================


def main() -> None:
    left_path = Path(LEFT_PANEL)
    right_path = Path(RIGHT_PANEL)
    if not left_path.is_file():
        raise FileNotFoundError(f"Left panel does not exist: {left_path}")
    if not right_path.is_file():
        raise FileNotFoundError(f"Right panel does not exist: {right_path}")

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")

    width = OUTER_PADDING * 2 + left.width + GAP + right.width
    height = OUTER_PADDING * 2 + max(left.height, right.height)
    canvas = Image.new("RGB", (width, height), "white")

    left_y = OUTER_PADDING + (height - 2 * OUTER_PADDING - left.height) // 2
    right_y = OUTER_PADDING + (height - 2 * OUTER_PADDING - right.height) // 2
    canvas.paste(left, (OUTER_PADDING, left_y))
    canvas.paste(right, (OUTER_PADDING + left.width + GAP, right_y))

    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    print(f"Saved combined left/right comparison figure to: {output}")


if __name__ == "__main__":
    main()
