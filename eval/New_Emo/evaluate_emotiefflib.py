#!/usr/bin/env python3
"""
EmotiEffLib emotion evaluation for generated videos.

Performs frame-level facial emotion recognition using EmotiEffLib (PyTorch backend)
on either a single video or a directory of videos.

Pipeline per video:
  1. Decode frames with OpenCV at the given stride.
  2. Detect faces with MTCNN (facenet-pytorch). Largest face is taken.
  3. Crop and align the face, resize to the model's input size.
  4. Call EmotiEffLib's `predict_emotions(cropped_face, logits=False)` to get
     per-frame softmax probabilities over 8 AffectNet classes.

If no face is detected on a given frame, that frame is skipped (counted as a
miss). Per-frame results are aggregated into per-video summary statistics
(dominant emotion, distribution, mean valence/arousal for MTL models).

Usage:
    # single video
    python evaluate_emotiefflib.py --video /path/to/video.mp4

    # single video with GT label
    python evaluate_emotiefflib.py --video /path/to/video.mp4 --label happiness

    # batch evaluation over a directory
    python evaluate_emotiefflib.py --video_dir /path/to/videos/

    # batch with GT labels from a manifest file (one "<stem> <label>" per line)
    python evaluate_emotiefflib.py --video_dir /path/to/videos/ --label_file labels.txt

    # choose model + device
    python evaluate_emotiefflib.py --video a.mp4 --model enet_b2_8 --device cuda:0

    # disable face detection and feed whole frames (NOT RECOMMENDED)
    python evaluate_emotiefflib.py --video a.mp4 --no_face_detect

    # save per-frame scores to json
    python evaluate_emotiefflib.py --video a.mp4 --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Suppress timm warnings on import (EmotiEffLib triggers harmless deprecation msgs).
os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _import_emotiefflib():
    """Import emotiefflib lazily so the script can still show --help without it."""
    try:
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list  # noqa
    except ImportError as exc:  # pragma: no cover
        sys.stderr.write(
            "ERROR: emotiefflib is not installed in this Python environment.\n"
            "Run:  pip install 'emotiefflib[torch]'\n"
            f"underlying error: {exc}\n"
        )
        sys.exit(2)
    return EmotiEffLibRecognizer, get_model_list


class FaceDetector:
    """Wrapper around facenet-pytorch MTCNN.

    The detector returns a list of bounding boxes per frame as
    ``[[x1, y1, x2, y2, prob], ...]`` in pixel coords. The recognizer
    uses the largest box (by area) as the primary face for that frame.
    """

    def __init__(self, device: str = "cuda"):
        from facenet_pytorch import MTCNN  # local import to avoid hard dep on --help
        # keep_all=False → return only the highest-confidence box
        # post_process=False → return boxes rather than aligned tensors
        self.mtcnn = MTCNN(
            keep_all=False,
            post_process=False,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            device=device,
        )

    def detect(self, rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        """Detect the largest face in `rgb` (H, W, 3 uint8). Returns (x1,y1,x2,y2) or None."""
        try:
            box, _ = self.mtcnn.detect(rgb)
        except Exception:
            return None
        if box is None:
            return None
        box = np.asarray(box, dtype=float)
        if box.ndim == 2 and box.shape[0] >= 1:
            # Multiple boxes: pick the largest by area. (keep_all=False should give 1 row,
            # but older facenet-pytorch may still return a 2-D array.)
            areas = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
            box = box[int(np.argmax(areas))]
        elif box.ndim == 1:
            pass  # already (4,)
        else:
            return None
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        # clamp to image bounds
        h, w = rgb.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2


def list_videos(path: Path) -> list[Path]:
    """Return a sorted list of video files under `path`."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in exts and p.is_file())


def load_label_map(args) -> dict[str, str]:
    """If a label_file is given, parse `<stem> <label>` per line into a dict."""
    if not args.label_file:
        return {}
    label_map: dict[str, str] = {}
    with open(args.label_file) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            stem, lbl = parts[0], " ".join(parts[1:])
            label_map[stem] = lbl
    return label_map


def video_frames(video_path: Path, frame_stride: int = 1, max_frames: int | None = None):
    """Yield BGR numpy frames from a video file. Stops after max_frames if given."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    idx = 0
    yielded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                yield idx, frame  # yield BGR; consumer handles color conversion
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def aggregate_emotions(frame_results: list[dict]) -> dict[str, Any]:
    """Compute summary statistics from a list of per-frame result dicts."""
    if not frame_results:
        return {"frames_analyzed": 0}
    counter = Counter(r["emotion"] for r in frame_results if r.get("emotion"))
    total = sum(counter.values())
    if total == 0:
        return {"frames_analyzed": len(frame_results), "dominant_emotion": None}
    distribution = {k: round(v / total, 4) for k, v in counter.most_common()}
    dominant_emotion, dominant_count = counter.most_common(1)[0]
    return {
        "frames_analyzed": len(frame_results),
        "frames_with_face": total,
        "dominant_emotion": dominant_emotion,
        "dominant_fraction": round(dominant_count / total, 4),
        "emotion_distribution": distribution,
    }


def evaluate_video(
    recognizer,
    video_path: Path,
    label: str | None,
    frame_stride: int,
    max_frames: int | None,
    use_face_detect: bool,
    face_detector: FaceDetector | None,
    show_progress: bool = False,
    is_mtl: bool = False,
    model_idx_to_emotion: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Run frame-level emotion recognition on a single video."""
    frame_records: list[dict] = []
    mtl_records: list[dict] = []

    iterator = video_frames(video_path, frame_stride=frame_stride, max_frames=max_frames)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=video_path.name, unit="frame")
        except ImportError:
            pass

    n_frames = 0
    n_with_face = 0
    for frame_idx, bgr in iterator:
        n_frames += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if use_face_detect:
            box = face_detector.detect(rgb) if face_detector is not None else None
            if box is None:
                frame_records.append({"frame_idx": frame_idx, "emotion": None,
                                      "score": None, "n_faces": 0,
                                      "valence": None, "arousal": None})
                continue
            x1, y1, x2, y2 = box
            face_rgb = rgb[y1:y2, x1:x2]
        else:
            face_rgb = rgb
            box = None

        # Run EmotiEffLib on the cropped face.
        # `predict_emotions` expects a face image (RGB uint8).
        try:
            emotions, scores = recognizer.predict_emotions([face_rgb], logits=False)
        except Exception as exc:
            frame_records.append({"frame_idx": frame_idx, "emotion": None,
                                  "score": None, "n_faces": 0,
                                  "valence": None, "arousal": None,
                                  "error": str(exc)})
            continue

        if not emotions or scores is None or len(emotions) == 0:
            frame_records.append({"frame_idx": frame_idx, "emotion": None,
                                  "score": None, "n_faces": 0,
                                  "valence": None, "arousal": None})
            continue

        emo = emotions[0]
        prob_row = np.asarray(scores[0], dtype=float)
        # For MTL models, scores has shape (n_classes+2): last two are valence/arousal.
        if is_mtl and prob_row.shape[-1] >= 10:
            valence = float(prob_row[-2])
            arousal = float(prob_row[-1])
            emo_probs = prob_row[:-2]
        else:
            valence = None
            arousal = None
            emo_probs = prob_row

        sc = float(np.max(emo_probs))
        rec = {"frame_idx": frame_idx, "emotion": emo, "score": round(sc, 4),
               "n_faces": 1, "valence": valence, "arousal": arousal}
        frame_records.append(rec)
        n_with_face += 1
        if valence is not None and arousal is not None:
            mtl_records.append({"frame_idx": frame_idx,
                                "valence": round(valence, 4),
                                "arousal": round(arousal, 4)})

    summary = aggregate_emotions(frame_records)
    summary["frames_total"] = n_frames
    summary["face_detection_rate"] = round(n_with_face / n_frames, 4) if n_frames else 0.0
    if mtl_records:
        summary["mean_valence"] = round(float(np.mean([r["valence"] for r in mtl_records])), 4)
        summary["mean_arousal"] = round(float(np.mean([r["arousal"] for r in mtl_records])), 4)

    result: dict[str, Any] = {
        "video": str(video_path),
        "model": args_model_name(recognizer),
        "label": label,
        "summary": summary,
        "frames": frame_records,
    }
    if label is not None and summary.get("dominant_emotion") is not None:
        result["correct"] = (label.lower() == summary["dominant_emotion"].lower())
    return result


def args_model_name(recognizer) -> str:
    """Best-effort name extraction for the loaded recognizer."""
    return getattr(recognizer, "model_name", None) or type(recognizer).__name__


def normalize_label_to_canonical(label: str) -> str | None:
    """Try to map user label (e.g. 'happy', 'sad') to EmotiEffLib class names."""
    if not label:
        return None
    s = label.strip().lower()
    # class names in the model: anger / contempt / disgust / fear / happiness / neutral / sadness / surprise
    synonyms = {
        "happy": "happiness",
        "happiness": "happiness",
        "sad": "sadness",
        "sadness": "sadness",
        "angry": "anger",
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral",
        "contempt": "contempt",
        "calm": "neutral",
    }
    return synonyms.get(s, s)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=str, help="Path to a single video file")
    p.add_argument("--video_dir", type=str, help="Path to a directory of videos")
    p.add_argument("--label_file", type=str,
                   help="Optional file with `<stem> <label>` per line for batch eval")
    p.add_argument("--label", type=str, help="Optional GT label for --video (single-video mode)")
    p.add_argument("--model", type=str, default="enet_b2_8",
                   help="EmotiEffLib model name (default: enet_b2_8). Run with --list_models to see options.")
    p.add_argument("--device", type=str, default="cuda",
                   help="Torch device, e.g. cuda:0, cpu (default: cuda)")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Analyze every Nth frame (default: 1 = every frame)")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Cap number of frames analyzed per video (for debugging)")
    p.add_argument("--no_face_detect", action="store_true",
                   help="Disable MTCNN face detection (feed whole frames; NOT RECOMMENDED, "
                        "model accuracy will be poor).")
    p.add_argument("--face_detector_device", type=str, default=None,
                   help="Override device for MTCNN (default: same as --device)")
    p.add_argument("--output", type=str, default=None,
                   help="Optional path to dump per-frame results as JSON")
    p.add_argument("--list_models", action="store_true", help="Print available model names and exit.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-video progress output")
    args = p.parse_args()

    EmotiEffLibRecognizer, get_model_list = _import_emotiefflib()

    if args.list_models:
        print("Available EmotiEffLib models:")
        for m in get_model_list():
            print(f"  - {m}")
        return

    if not args.video and not args.video_dir:
        p.error("Provide --video <path> or --video_dir <dir>")

    if args.model not in get_model_list():
        sys.stderr.write(f"ERROR: model {args.model!r} not supported. "
                         f"Choices: {get_model_list()}\n")
        sys.exit(2)

    if args.video:
        video_paths = [Path(args.video)]
        label_map = {}
        if args.label:
            label_map[video_paths[0].stem] = args.label
    else:
        video_paths = list_videos(Path(args.video_dir))
        label_map = load_label_map(args)
    if not video_paths:
        sys.stderr.write("ERROR: no videos found\n")
        sys.exit(2)

    print(f"[EmotiEffLib] loading model: {args.model} (device={args.device})", flush=True)
    recognizer = EmotiEffLibRecognizer(model_name=args.model, device=args.device)

    face_detector = None
    if not args.no_face_detect:
        det_device = args.face_detector_device or args.device
        print(f"[EmotiEffLib] loading MTCNN face detector on {det_device}", flush=True)
        face_detector = FaceDetector(device=det_device)

    results: list[dict] = []
    n_correct = 0
    n_labelled = 0
    for i, vp in enumerate(video_paths, 1):
        lbl = label_map.get(vp.stem)
        if not args.quiet:
            print(f"[{i}/{len(video_paths)}] {vp}", flush=True)
        try:
            r = evaluate_video(
                recognizer, vp, lbl, args.frame_stride, args.max_frames,
                use_face_detect=not args.no_face_detect,
                face_detector=face_detector,
                show_progress=not args.quiet,
                is_mtl=False,  # default enet_b2_8 has 8 classes
                model_idx_to_emotion=None,
            )
        except Exception as exc:
            print(f"  !! failed: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc()
            r = {"video": str(vp), "model": args.model, "label": lbl, "error": str(exc)}
        results.append(r)
        if lbl is not None and r.get("correct"):
            n_correct += 1
        if lbl is not None:
            n_labelled += 1
        s = r.get("summary", {})
        dom = s.get("dominant_emotion")
        print(f"  -> dominant={dom!r} dist={s.get('emotion_distribution', {})} "
              f"correct={r.get('correct')}", flush=True)

    overall = {
        "model": args.model,
        "device": args.device,
        "face_detection": not args.no_face_detect,
        "n_videos": len(results),
        "n_labelled": n_labelled,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n_labelled, 4) if n_labelled else None,
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(overall, fh, indent=2)
        print(f"[EmotiEffLib] wrote {args.output}")
    else:
        print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()