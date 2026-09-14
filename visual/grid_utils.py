"""Video-grid visualization helpers used by the paper figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    duration: float


def probe_video(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 25.0
    if frame_count <= 0:
        raise RuntimeError(f"Cannot read frame count from video: {path}")

    duration = frame_count / fps
    return VideoInfo(path=path, fps=fps, frame_count=frame_count, duration=duration)


def make_common_sample_times(
    infos: Sequence[VideoInfo],
    num_samples: int,
    start_ratio: float = 0.10,
    end_ratio: float = 0.90,
) -> np.ndarray:
    if not infos:
        raise ValueError("No videos were provided.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    if not (0.0 <= start_ratio <= end_ratio <= 1.0):
        raise ValueError("Require 0 <= start_ratio <= end_ratio <= 1.")

    # Use the shortest video as the common temporal support, so the k-th sample
    # represents the same absolute timestamp across all compared videos.
    common_duration = min(info.duration for info in infos)
    if common_duration <= 0:
        raise RuntimeError("At least one input video has zero duration.")

    max_time = max(0.0, common_duration - 1.0 / max(info.fps for info in infos))
    start_time = start_ratio * max_time
    end_time = end_ratio * max_time
    if num_samples == 1:
        return np.asarray([(start_time + end_time) * 0.5], dtype=np.float64)
    return np.linspace(start_time, end_time, num_samples, dtype=np.float64)


def read_frame_at_time(info: VideoInfo, time_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {info.path}")

    # Timestamp seeking is preferable when comparison videos have different FPS.
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
    ok, frame = cap.read()

    # Some codecs/backends have unreliable timestamp seeking. Fall back to the
    # corresponding frame index in that case.
    if not ok or frame is None:
        frame_index = min(
            info.frame_count - 1,
            max(0, int(round(float(time_sec) * info.fps))),
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()

    cap.release()
    if not ok or frame is None:
        raise RuntimeError(
            f"Failed to decode {info.path} at t={float(time_sec):.3f}s."
        )
    return frame


def letterbox_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Cell width/height must be positive.")

    src_h, src_w = frame.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError("Invalid frame shape.")

    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _load_font(font_path: str | None, font_size: int) -> ImageFont.ImageFont:
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(["DejaVuSans.ttf", "Arial.ttf"])

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xyxy: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    x0, y0, x1, y1 = xyxy
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - tw) / 2.0
    y = y0 + (y1 - y0 - th) / 2.0 - bbox[1]
    draw.text((x, y), text, font=font, fill=fill)


def build_video_grid(
    videos: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    num_samples: int = 5,
    start_ratio: float = 0.10,
    end_ratio: float = 0.90,
    cell_width: int = 256,
    cell_height: int = 256,
    label_axis: str = "x",
    label_size: int = 30,
    label_band: int = 54,
    gap: int = 4,
    outer_padding: int = 8,
    border_width: int = 1,
    font_path: str | None = None,
    panel_title: str | None = None,
    title_size: int = 32,
    title_band: int = 54,
) -> Path:
    """Build a paper-style video frame grid and save it as a lossless PNG."""
    if not videos:
        raise ValueError("videos mapping is empty.")
    if label_axis not in {"x", "y"}:
        raise ValueError("label_axis must be 'x' or 'y'.")

    labels = list(videos.keys())
    infos = [probe_video(path) for path in videos.values()]
    times = make_common_sample_times(infos, num_samples, start_ratio, end_ratio)

    frames_by_label: list[list[np.ndarray]] = []
    for info in infos:
        frames = [
            letterbox_bgr(read_frame_at_time(info, t), cell_width, cell_height)
            for t in times
        ]
        frames_by_label.append(frames)

    title_h = title_band if panel_title else 0
    if label_axis == "x":
        grid_w = len(labels) * cell_width + max(0, len(labels) - 1) * gap
        grid_h = num_samples * cell_height + max(0, num_samples - 1) * gap
        canvas_w = outer_padding * 2 + grid_w
        canvas_h = outer_padding * 2 + title_h + label_band + grid_h
        grid_x0 = outer_padding
        grid_y0 = outer_padding + title_h + label_band
    else:
        grid_w = num_samples * cell_width + max(0, num_samples - 1) * gap
        grid_h = len(labels) * cell_height + max(0, len(labels) - 1) * gap
        canvas_w = outer_padding * 2 + label_band * 2 + grid_w
        canvas_h = outer_padding * 2 + title_h + grid_h
        grid_x0 = outer_padding + label_band * 2
        grid_y0 = outer_padding + title_h

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    if label_axis == "x":
        for col, _label in enumerate(labels):
            x = grid_x0 + col * (cell_width + gap)
            for row in range(num_samples):
                y = grid_y0 + row * (cell_height + gap)
                canvas[y : y + cell_height, x : x + cell_width] = frames_by_label[col][row]
    else:
        for row, _label in enumerate(labels):
            y = grid_y0 + row * (cell_height + gap)
            for col in range(num_samples):
                x = grid_x0 + col * (cell_width + gap)
                canvas[y : y + cell_height, x : x + cell_width] = frames_by_label[row][col]

    # Draw cell borders after frame placement.
    if border_width > 0:
        if label_axis == "x":
            for col in range(len(labels)):
                for row in range(num_samples):
                    x = grid_x0 + col * (cell_width + gap)
                    y = grid_y0 + row * (cell_height + gap)
                    cv2.rectangle(
                        canvas,
                        (x, y),
                        (x + cell_width - 1, y + cell_height - 1),
                        (20, 20, 20),
                        border_width,
                    )
        else:
            for row in range(len(labels)):
                for col in range(num_samples):
                    x = grid_x0 + col * (cell_width + gap)
                    y = grid_y0 + row * (cell_height + gap)
                    cv2.rectangle(
                        canvas,
                        (x, y),
                        (x + cell_width - 1, y + cell_height - 1),
                        (20, 20, 20),
                        border_width,
                    )

    # PIL gives substantially cleaner anti-aliased text than cv2.putText.
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    label_font = _load_font(font_path, label_size)
    title_font = _load_font(font_path, title_size)

    if panel_title:
        _draw_centered_text(
            draw,
            (outer_padding, outer_padding, canvas_w - outer_padding, outer_padding + title_h),
            panel_title,
            title_font,
        )

    if label_axis == "x":
        y0 = outer_padding + title_h
        for col, label in enumerate(labels):
            x0 = grid_x0 + col * (cell_width + gap)
            _draw_centered_text(
                draw,
                (x0, y0, x0 + cell_width, y0 + label_band),
                label,
                label_font,
            )
    else:
        for row, label in enumerate(labels):
            y0 = grid_y0 + row * (cell_height + gap)
            _draw_centered_text(
                draw,
                (outer_padding, y0, outer_padding + label_band * 2, y0 + cell_height),
                label,
                label_font,
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
    return output
