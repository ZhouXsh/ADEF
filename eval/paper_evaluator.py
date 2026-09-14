#!/usr/bin/env python3
"""Authoritative paper-table evaluator for ADEF and all baselines.

Protocol v4 uses a record-first evaluation rule:
- retain one record per video with frame counts, metric values and participation;
- aggregate paper-facing scalar metrics only after the corresponding video-level
  records are complete;
- keep FID/FVD as standard dataset-level Frechet metrics over successful pairs.

Completed metric outputs are persistent. A metric is skipped when its cached
output matches the current input files, protocol and metric-specific config.

Sample failures are excluded only from the affected metric aggregate. All
failures remain explicit in ``failed_samples.csv``.

Device semantics:
- ``--device cuda:N`` means physical GPU N at this entry point;
- metric subprocesses are launched with ``CUDA_VISIBLE_DEVICES=N`` and receive
  logical ``cuda:0`` so nested PyTorch/TensorFlow subprocesses stay on that GPU;
- ``--device cpu`` hides CUDA from all metric subprocesses.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from paper_protocol import (  # noqa: E402
    DFER_CLIP_EMOTIONS,
    PROTOCOL_VERSION,
    Sample,
    manifest_fingerprint,
    read_manifest,
)
from paper_table_utils import trim_paper_table  # noqa: E402

LSE_SCRIPT = THIS_DIR / "Wav2Lip" / "evaluation" / "eval_lipsync.py"
FID_SCRIPT = THIS_DIR / "pytorch-fid" / "evaluate_fid_video.py"
FVD_SCRIPT = THIS_DIR / "frechet_video_distance" / "evaluate_adef.py"
PAIRWISE_SCRIPT = THIS_DIR / "pairwise_metrics.py"
EMOTIEFF_SCRIPT = THIS_DIR / "New_Emo" / "evaluate_emotiefflib.py"
DFER_SCRIPT = THIS_DIR / "New_Emo" / "evaluate_dfer_clip.py"

DEFAULT_EVAL_PY = Path("/home/Zhouxishi/miniconda3/envs/eval/bin/python")
DEFAULT_FVD_PY = Path("/home/Zhouxishi/miniconda3/envs/fvd/bin/python")
DEFAULT_LSE_PY = THIS_DIR / "Wav2Lip" / "evaluation" / "venv" / "bin" / "python"
DEFAULT_PAIRWISE_PY = THIS_DIR / "evaluation_eat" / "venv" / "bin" / "python"

PAPER_METRICS = ("lse", "fid", "fvd", "pairwise", "emotiefflib", "dfer_clip")
TABLE_COLUMNS = [
    "Method", "Status", "N", "Evaluated-N",
    "LSE-D", "LSE-C", "LSE-N",
    "FID", "FID-N", "FVD", "FVD-N",
    "PSNR", "PSNR-N", "SSIM", "SSIM-N", "LPIPS", "LPIPS-N",
    "M-LMD", "F-LMD", "LMD-N",
    "EmotiEff-Acc", "EmotiEff-N", "DFER-CLIP-Acc", "DFER-N",
    "Protocol", "Manifest-SHA256",
]
FAILURE_COLUMNS = ["metric", "name", "fake", "gt", "error"]
CACHE_VERSION = 1


def _python(preferred: Path) -> str:
    return str(preferred) if preferred.is_file() else sys.executable


def _to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _device_context(device: str) -> tuple[str, dict[str, str], dict[str, Any]]:
    requested = str(device or "cuda:0").strip().lower()
    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if requested == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu", env, {"requested": requested, "effective": "cpu", "cuda_visible_devices": ""}
    if requested in {"cuda", "gpu"}:
        visible = env.get("CUDA_VISIBLE_DEVICES")
        return "cuda:0", env, {"requested": requested, "effective": "cuda:0", "cuda_visible_devices": visible}
    match = re.fullmatch(r"(?:cuda|gpu):(\d+)", requested)
    if not match:
        raise ValueError(f"invalid --device {device!r}; expected cpu, cuda, or cuda:N")
    physical = match.group(1)
    env["CUDA_VISIBLE_DEVICES"] = physical
    return "cuda:0", env, {
        "requested": requested, "effective": "cuda:0", "physical_gpu": int(physical),
        "cuda_visible_devices": physical,
    }


def _run(cmd, *, cwd=None, timeout: int | None = 7200, env=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True,
                           text=True, timeout=timeout, env=env)
        return {"rc": p.returncode, "stdout": _to_text(p.stdout), "stderr": _to_text(p.stderr),
                "elapsed_sec": time.time() - t0, "cmd": [str(x) for x in cmd], "cached": False}
    except subprocess.TimeoutExpired as exc:
        return {"rc": 124, "stdout": _to_text(exc.stdout),
                "stderr": f"timeout after {timeout}s" + (f"\n{_to_text(exc.stderr)}" if exc.stderr else ""),
                "elapsed_sec": time.time() - t0, "cmd": [str(x) for x in cmd], "cached": False}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_present(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, f"output missing: {path}"
    try:
        return _load_json(path), None
    except Exception as exc:
        return None, f"invalid JSON {path}: {type(exc).__name__}: {exc}"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _proc_tail(proc: dict) -> str:
    lines = (proc.get("stderr") or proc.get("stdout") or "").splitlines()
    return " | ".join(lines[-6:])


def _write_list(path: Path, values: list[str]) -> Path:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")
    return path


def _safe_stem(name: str, index: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-") or "sample"
    return f"{index:05d}_{safe}"


def _emotion_stem_map(samples: list[Sample]) -> dict[str, Sample]:
    return {_safe_stem(s.name, i): s for i, s in enumerate(samples)}


def _stage_emotion_inputs(samples: list[Sample], root: Path):
    if root.is_symlink() or root.is_file():
        root.unlink()
    elif root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    labels = root / "labels.txt"
    stem_to_sample = _emotion_stem_map(samples)
    with labels.open("w", encoding="utf-8") as lf:
        for i, s in enumerate(samples):
            stem = _safe_stem(s.name, i)
            dst = root / f"{stem}{Path(s.fake).suffix.lower() or '.mp4'}"
            src = Path(s.fake).resolve()
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            if s.emotion:
                lf.write(f"{dst.stem} {s.emotion}\n")
    return labels, stem_to_sample


def _mean(payload: dict, key: str):
    value = payload.get(key)
    return value.get("mean") if isinstance(value, dict) else None


def _failure(metric: str, sample: Sample | None, error: str, **extra) -> dict[str, Any]:
    row = {
        "metric": metric,
        "name": sample.name if sample else extra.pop("name", ""),
        "fake": sample.fake if sample else extra.pop("fake", ""),
        "gt": sample.gt if sample else extra.pop("gt", ""),
        "error": error,
    }
    row.update(extra)
    return row


def _read_upstream_failures(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return [_failure("input", None, f"upstream failure file missing: {p}")]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return [_failure("input", None, f"cannot read upstream failures: {type(exc).__name__}: {exc}")]
    rows = data.get("failures", []) if isinstance(data, dict) else data
    out = []
    for raw in rows if isinstance(rows, list) else []:
        if isinstance(raw, dict):
            out.append({"metric": str(raw.get("metric") or raw.get("stage") or "input"),
                        "name": str(raw.get("name") or ""), "fake": str(raw.get("fake") or ""),
                        "gt": str(raw.get("gt") or ""),
                        "error": str(raw.get("error") or "upstream sample failure")})
    return out


def _sample_by_index(samples: list[Sample], index: Any) -> Sample | None:
    try:
        i = int(index)
    except (TypeError, ValueError):
        return None
    return samples[i] if 0 <= i < len(samples) else None


def _add_global_failure(failures: list[dict], metric: str, error: str) -> None:
    failures.append(_failure(metric, None, error))


def _norm_path(value: str | None) -> str:
    return str(Path(value or "").expanduser().resolve())


def _file_state(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        st = p.stat()
        return {"path": str(p.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    except OSError:
        return {"path": str(p), "size": None, "mtime_ns": None}


def _input_state(samples: list[Sample]) -> list[dict[str, Any]]:
    return [{"name": s.name, "emotion": s.emotion,
             "fake": _file_state(s.fake), "gt": _file_state(s.gt)} for s in samples]


def _metric_script(metric: str) -> Path:
    return {
        "lse": LSE_SCRIPT, "fid": FID_SCRIPT, "fvd": FVD_SCRIPT,
        "pairwise": PAIRWISE_SCRIPT, "emotiefflib": EMOTIEFF_SCRIPT, "dfer_clip": DFER_SCRIPT,
    }[metric]


def _metric_config(metric: str, args) -> dict[str, Any]:
    if metric == "lse":
        return {"min_track": args.lse_min_track}
    if metric == "fid":
        return {"frame_stride": args.fid_frame_stride}
    if metric == "fvd":
        return {"video_length": args.fvd_video_length}
    if metric == "pairwise":
        return {"metrics": ["psnr", "ssim", "lpips", "lmd"], "alignment": "EAT", "lpips_net": "alex"}
    if metric == "emotiefflib":
        return {"model": args.emotieff_model, "frame_stride": 1}
    if metric == "dfer_clip":
        return {"num_segments": args.dfer_segments}
    raise KeyError(metric)


def _metric_fingerprint(samples: list[Sample], metric: str, args) -> str:
    return manifest_fingerprint(samples, [metric], context={
        "cache_version": CACHE_VERSION,
        "config": _metric_config(metric, args),
        "input_state": _input_state(samples),
        "script_state": _file_state(_metric_script(metric)),
    })


def _metric_output_complete(metric: str, data: dict, samples: list[Sample]) -> bool:
    n = len(samples)
    if metric == "lse":
        return data.get("n_total") == n and len(data.get("results", [])) == n
    if metric in {"fid", "fvd"}:
        return data.get(metric) is not None and not data.get("global_error") and len(data.get("per_video", [])) == n
    if metric == "pairwise":
        return bool(data.get("complete")) and data.get("n_samples") == n and data.get("processed_n") == n
    if metric in {"emotiefflib", "dfer_clip"}:
        return data.get("n_videos") == n and len(data.get("results", [])) == n
    return False


def _metric_output_matches_config(metric: str, data: dict, args) -> bool:
    if metric == "lse":
        return (data.get("config") or {}).get("min_track") == args.lse_min_track
    if metric == "fid":
        return (data.get("config") or {}).get("frame_stride") == args.fid_frame_stride
    if metric == "fvd":
        return data.get("video_length") == args.fvd_video_length
    if metric == "pairwise":
        return data.get("protocol_version") == PROTOCOL_VERSION and data.get("pairwise_aggregation_version") is not None
    if metric == "emotiefflib":
        return data.get("model") == args.emotieff_model
    if metric == "dfer_clip":
        return data.get("num_segments") == args.dfer_segments
    return False


def _metric_output_matches_samples(metric: str, data: dict, samples: list[Sample]) -> bool:
    try:
        if metric == "lse":
            got = [_norm_path(r.get("video")) for r in data.get("results", [])]
            return got == [_norm_path(s.fake) for s in samples]
        if metric in {"fid", "fvd"}:
            rows = data.get("per_video", [])
            if len(rows) != len(samples):
                return False
            return all(_norm_path(r.get("real")) == _norm_path(s.gt) and
                       _norm_path(r.get("fake")) == _norm_path(s.fake)
                       for r, s in zip(rows, samples))
        if metric == "pairwise":
            rows = data.get("per_video", [])
            if len(rows) != len(samples):
                return False
            return all(r.get("name") == s.name and _norm_path(r.get("fake")) == _norm_path(s.fake) and
                       _norm_path(r.get("gt")) == _norm_path(s.gt) for r, s in zip(rows, samples))
        if metric in {"emotiefflib", "dfer_clip"}:
            got = [Path(r.get("video", "")).stem for r in data.get("results", [])]
            return got == [_safe_stem(s.name, i) for i, s in enumerate(samples)]
    except Exception:
        return False
    return False


def _cache_meta_path(out: Path) -> Path:
    return out.with_name(out.stem + ".cache.json")


def _legacy_output_is_safe(metric: str, out: Path, samples: list[Sample]) -> bool:
    try:
        output_mtime = out.stat().st_mtime_ns
        input_mtime = max([Path(s.fake).stat().st_mtime_ns for s in samples] +
                          [Path(s.gt).stat().st_mtime_ns for s in samples])
        if output_mtime < input_mtime:
            return False
        if metric != "pairwise" and output_mtime < _metric_script(metric).stat().st_mtime_ns:
            return False
        return True
    except OSError:
        return False


def _load_completed_metric(metric: str, out: Path, samples: list[Sample], args) -> tuple[dict | None, str | None]:
    data, _ = _load_json_if_present(out)
    if data is None or not _metric_output_complete(metric, data, samples):
        return None, None
    if not _metric_output_matches_config(metric, data, args) or not _metric_output_matches_samples(metric, data, samples):
        return None, None

    fingerprint = _metric_fingerprint(samples, metric, args)
    meta_path = _cache_meta_path(out)
    meta, _ = _load_json_if_present(meta_path)
    if meta is not None:
        if meta.get("cache_version") == CACHE_VERSION and meta.get("complete") is True and meta.get("fingerprint") == fingerprint:
            return data, "validated cache"
        return None, None

    if not _legacy_output_is_safe(metric, out, samples):
        return None, None
    _atomic_json(meta_path, {
        "cache_version": CACHE_VERSION, "metric": metric, "complete": True,
        "fingerprint": fingerprint, "adopted_existing_output": True,
        "recorded_at": time.time(),
    })
    return data, "adopted existing completed output"


def _record_metric_cache(metric: str, out: Path, samples: list[Sample], args, data: dict) -> None:
    if not _metric_output_complete(metric, data, samples):
        return
    if not _metric_output_matches_config(metric, data, args) or not _metric_output_matches_samples(metric, data, samples):
        return
    _atomic_json(_cache_meta_path(out), {
        "cache_version": CACHE_VERSION, "metric": metric, "complete": True,
        "fingerprint": _metric_fingerprint(samples, metric, args),
        "recorded_at": time.time(),
    })


def _run_metric(metric: str, out: Path, samples: list[Sample], args,
                command_factory: Callable[[], tuple[list[str], Path]], *, timeout: int | None,
                env: dict[str, str]) -> tuple[dict | None, str | None, dict]:
    force = metric in set(args.force_metrics or [])
    if not force:
        cached, why = _load_completed_metric(metric, out, samples, args)
        if cached is not None:
            print(f"[paper-eval] SKIP {metric}: {why}", flush=True)
            return cached, None, {"rc": 0, "cached": True, "elapsed_sec": 0.0,
                                  "stdout": "", "stderr": "", "cmd": []}
    print(f"[paper-eval] RUN  {metric}" + (" (forced)" if force else ""), flush=True)
    out.unlink(missing_ok=True)
    cmd, cwd = command_factory()
    proc = _run(cmd, cwd=cwd, timeout=timeout, env=env)
    data, load_err = _load_json_if_present(out)
    if data is not None:
        _record_metric_cache(metric, out, samples, args, data)
    return data, load_err, proc


def evaluate(samples: list[Sample], method: str, outdir: Path, metrics: list[str], args) -> tuple[dict, dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "work"
    work.mkdir(parents=True, exist_ok=True)
    expected_n = args.expected_n if args.expected_n is not None else len(samples)
    if expected_n < len(samples):
        raise ValueError(f"--expected-n {expected_n} is smaller than manifest size {len(samples)}")

    metric_device, metric_env, device_info = _device_context(args.device)
    print(f"[paper-eval] device requested={args.device} effective={metric_device} "
          f"CUDA_VISIBLE_DEVICES={device_info.get('cuda_visible_devices')!r}", flush=True)

    details: dict[str, Any] = {}
    failures: list[dict[str, Any]] = _read_upstream_failures(args.upstream_failures)
    hard_errors: list[str] = []
    coverage: dict[str, int | None] = {}
    per_video = {s.name: {"name": s.name, "fake": s.fake, "gt": s.gt, "emotion": s.emotion} for s in samples}
    fake_list = _write_list(work / "fake.txt", [s.fake for s in samples])
    gt_list = _write_list(work / "gt.txt", [s.gt for s in samples])

    if "lse" in metrics:
        out = work / "lse.json"
        data, load_err, proc = _run_metric(
            "lse", out, samples, args,
            lambda: ([args.lse_python, str(LSE_SCRIPT), "--filelist", str(fake_list), "--output_json", str(out),
                      "--device", metric_device, "--min-track", str(args.lse_min_track)], LSE_SCRIPT.parent),
            timeout=args.timeout, env=metric_env)
        if data is None:
            msg = f"LSE unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "LSE", msg)
            details["lse"] = {"process": proc, "error": msg}; coverage["lse"] = 0
        else:
            data["process"] = proc; details["lse"] = data
            n_ok = 0; results = data.get("results", [])
            for i, s in enumerate(samples):
                r = results[i] if i < len(results) else {"error": "missing LSE result row"}
                per_video[s.name].update({"LSE-D": r.get("lse_d"), "LSE-C": r.get("lse_c")})
                if r.get("error") or r.get("lse_d") is None or r.get("lse_c") is None:
                    failures.append(_failure("LSE", s, str(r.get("error") or "missing LSE score")))
                else:
                    n_ok += 1
            coverage["lse"] = n_ok
            if n_ok == 0:
                msg = "LSE has zero successful samples"; hard_errors.append(msg); _add_global_failure(failures, "LSE", msg)

    if "fid" in metrics:
        out = work / "fid.json"
        def fid_cmd():
            cmd = [args.eval_python, str(FID_SCRIPT), "--list1", str(gt_list), "--list2", str(fake_list),
                   "--output-json", str(out), "--device", metric_device]
            if args.fid_frame_stride != 1:
                cmd += ["--frame-stride", str(args.fid_frame_stride)]
            return cmd, FID_SCRIPT.parent
        data, load_err, proc = _run_metric("fid", out, samples, args, fid_cmd, timeout=args.timeout, env=metric_env)
        if data is None:
            msg = f"FID unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "FID", msg)
            details["fid"] = {"process": proc, "error": msg}; coverage["fid"] = 0
        else:
            data["process"] = proc; details["fid"] = data
            coverage["fid"] = int(data.get("n_success") if data.get("n_success") is not None else len(samples))
            for f in data.get("failures", []):
                s = _sample_by_index(samples, f.get("index"))
                failures.append(_failure("FID", s, str(f.get("error") or "FID pair failed"),
                                         name=f.get("fake", "") if s is None else ""))
            if data.get("fid") is None:
                msg = str(data.get("global_error") or "FID has no usable value")
                hard_errors.append(msg); _add_global_failure(failures, "FID", msg)

    if "fvd" in metrics:
        out = work / "fvd.json"
        data, load_err, proc = _run_metric(
            "fvd", out, samples, args,
            lambda: ([args.fvd_python, str(FVD_SCRIPT), "--real_list", str(gt_list), "--fake_list", str(fake_list),
                      "--video_length", str(args.fvd_video_length), "--output_file", str(out),
                      "--device", metric_device], FVD_SCRIPT.parent),
            timeout=args.timeout, env=metric_env)
        if data is None:
            msg = f"FVD unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "FVD", msg)
            details["fvd"] = {"process": proc, "error": msg}; coverage["fvd"] = 0
        else:
            data["process"] = proc; details["fvd"] = data
            coverage["fvd"] = int(data.get("n_success") if data.get("n_success") is not None else data.get("num_videos", 0))
            for f in data.get("failures", []):
                failures.append(_failure("FVD", _sample_by_index(samples, f.get("index")),
                                         str(f.get("error") or "FVD pair failed")))
            if data.get("fvd") is None:
                msg = str(data.get("global_error") or "FVD has no usable value")
                hard_errors.append(msg); _add_global_failure(failures, "FVD", msg)

    if "pairwise" in metrics:
        manifest = work / "pairwise_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["name", "fake", "gt", "emotion"])
            for s in samples:
                w.writerow([s.name, s.fake, s.gt, s.emotion or ""])
        out = work / "pairwise.json"
        data, load_err, proc = _run_metric(
            "pairwise", out, samples, args,
            lambda: ([args.pairwise_python, str(PAIRWISE_SCRIPT), "--manifest", str(manifest),
                      "--output", str(out), "--device", metric_device], THIS_DIR),
            timeout=(args.pairwise_timeout if args.pairwise_timeout > 0 else None), env=metric_env)
        if data is None:
            msg = f"Pairwise unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "pairwise", msg)
            details["pairwise"] = {"process": proc, "error": msg}
            for key in ("psnr", "ssim", "lpips", "lmd"):
                coverage[key] = 0
        else:
            data["process"] = proc; details["pairwise"] = data
            cov = data.get("coverage", {})
            for key in ("psnr", "ssim", "lpips", "lmd"):
                coverage[key] = int(cov.get(key) or 0)
            for f in data.get("failures", []):
                name = str(f.get("name") or "")
                s = next((x for x in samples if x.name == name), None)
                metric = str(f.get("metric") or "pairwise")
                failures.append(_failure(metric.upper() if metric != "lmd" else "LMD", s,
                                         str(f.get("error") or "pairwise sample failed"), name=name if s is None else ""))
            for r in data.get("per_video", []):
                if r.get("name") in per_video:
                    per_video[r["name"]].update({"PSNR": r.get("psnr"), "SSIM": r.get("ssim"),
                                                  "LPIPS": r.get("lpips"), "M-LMD": r.get("mouth_lmd"),
                                                  "F-LMD": r.get("face_lmd")})
            agg = data.get("aggregate", {})
            for key, value in {"psnr": _mean(agg, "psnr"), "ssim": _mean(agg, "ssim"),
                               "lpips": _mean(agg, "lpips"), "lmd": _mean(agg, "mouth_lmd")}.items():
                if value is None:
                    msg = f"pairwise metric {key} has no usable value"
                    hard_errors.append(msg); _add_global_failure(failures, key.upper(), msg)

    emotion_stage: Path | None = None
    label_file: Path | None = None
    stem_map = _emotion_stem_map(samples)
    def ensure_emotion_stage() -> tuple[Path, dict[str, Sample]]:
        nonlocal emotion_stage, label_file, stem_map
        if label_file is None:
            emotion_stage = work / "emotion_inputs"
            label_file, stem_map = _stage_emotion_inputs(samples, emotion_stage)
        return label_file, stem_map

    if "emotiefflib" in metrics:
        out = work / "emotiefflib.json"
        def emot_cmd():
            labels, _ = ensure_emotion_stage()
            return ([args.eval_python, str(EMOTIEFF_SCRIPT), "--video_dir", str(emotion_stage),
                     "--label_file", str(labels), "--model", args.emotieff_model,
                     "--device", metric_device, "--quiet", "--output", str(out)], EMOTIEFF_SCRIPT.parent)
        data, load_err, proc = _run_metric("emotiefflib", out, samples, args, emot_cmd,
                                           timeout=args.timeout, env=metric_env)
        if data is None:
            msg = f"EmotiEff unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "EmotiEff", msg)
            details["emotiefflib"] = {"process": proc, "error": msg}; coverage["emotieff"] = 0
        else:
            successful_labelled = correct = 0
            for r in data.get("results", []):
                stem = Path(r.get("video", "")).stem
                s = stem_map.get(stem)
                pred = (r.get("summary") or {}).get("dominant_emotion")
                label = s.emotion if s else r.get("label")
                error = r.get("error")
                if error or pred is None or label is None:
                    reason = str(error or ("no valid dominant emotion" if pred is None else "missing target emotion label"))
                    failures.append(_failure("EmotiEff", s, reason, name=stem if s is None else ""))
                else:
                    successful_labelled += 1
                    is_correct = str(pred).lower() == str(label).lower()
                    correct += int(is_correct); r["correct_v4"] = is_correct
                if s:
                    per_video[s.name]["EmotiEff-Correct"] = r.get("correct_v4")
                    per_video[s.name]["EmotiEff-Pred"] = pred
            data["accuracy_v4"] = correct / successful_labelled if successful_labelled else None
            data["n_success_v4"] = successful_labelled; data["process"] = proc
            details["emotiefflib"] = data; coverage["emotieff"] = successful_labelled
            if successful_labelled == 0:
                msg = "EmotiEff has zero successfully labelled samples"
                hard_errors.append(msg); _add_global_failure(failures, "EmotiEff", msg)

    if "dfer_clip" in metrics:
        out = work / "dfer_clip.json"
        def dfer_cmd():
            labels, _ = ensure_emotion_stage()
            return ([args.eval_python, str(DFER_SCRIPT), "--video_dir", str(emotion_stage),
                     "--label_file", str(labels), "--device", metric_device,
                     "--num_segments", str(args.dfer_segments), "--quiet", "--output", str(out)], DFER_SCRIPT.parent)
        data, load_err, proc = _run_metric("dfer_clip", out, samples, args, dfer_cmd,
                                           timeout=args.timeout, env=metric_env)
        if data is None:
            msg = f"DFER-CLIP unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "DFER-CLIP", msg)
            details["dfer_clip"] = {"process": proc, "error": msg}; coverage["dfer"] = 0
        else:
            successful_supported = correct = 0
            for r in data.get("results", []):
                stem = Path(r.get("video", "")).stem
                s = stem_map.get(stem)
                label = s.emotion if s else r.get("label")
                supported = label in DFER_CLIP_EMOTIONS if label is not None else False
                pred = r.get("prediction")
                if label is None:
                    failures.append(_failure("DFER-CLIP", s, "missing target emotion label", name=stem if s is None else ""))
                    if s: per_video[s.name]["DFER-Label-Supported"] = False
                    continue
                if not supported:
                    if s: per_video[s.name]["DFER-Label-Supported"] = False
                    continue
                if r.get("error") or pred is None:
                    failures.append(_failure("DFER-CLIP", s, str(r.get("error") or "no prediction"),
                                             name=stem if s is None else ""))
                else:
                    successful_supported += 1
                    is_correct = str(pred).lower() == str(label).lower()
                    correct += int(is_correct); r["correct_v4"] = is_correct
                if s:
                    per_video[s.name]["DFER-Correct"] = r.get("correct_v4")
                    per_video[s.name]["DFER-Pred"] = pred
                    per_video[s.name]["DFER-Label-Supported"] = True
            data["accuracy_v4"] = correct / successful_supported if successful_supported else None
            data["n_success_supported_v4"] = successful_supported; data["process"] = proc
            details["dfer_clip"] = data; coverage["dfer"] = successful_supported
            eligible = sum(s.emotion in DFER_CLIP_EMOTIONS for s in samples)
            if eligible > 0 and successful_supported == 0:
                msg = "DFER-CLIP has zero successful supported-label samples"
                hard_errors.append(msg); _add_global_failure(failures, "DFER-CLIP", msg)

    lse = details.get("lse", {})
    pair = details.get("pairwise", {}).get("aggregate", {}) if isinstance(details.get("pairwise"), dict) else {}
    emot = details.get("emotiefflib", {})
    dfer = details.get("dfer_clip", {})
    status = "failed" if hard_errors else ("partial" if failures or expected_n != len(samples) else "complete")
    upstream_for_hash = _read_upstream_failures(args.upstream_failures)
    row = {
        "Method": method, "Status": status, "N": expected_n, "Evaluated-N": len(samples),
        "LSE-D": (((lse.get("aggregate") or {}).get("lse_d") or {}).get("mean")) if lse else None,
        "LSE-C": (((lse.get("aggregate") or {}).get("lse_c") or {}).get("mean")) if lse else None,
        "LSE-N": coverage.get("lse"),
        "FID": details.get("fid", {}).get("fid") if isinstance(details.get("fid"), dict) else None,
        "FID-N": coverage.get("fid"),
        "FVD": details.get("fvd", {}).get("fvd") if isinstance(details.get("fvd"), dict) else None,
        "FVD-N": coverage.get("fvd"),
        "PSNR": _mean(pair, "psnr"), "PSNR-N": coverage.get("psnr"),
        "SSIM": _mean(pair, "ssim"), "SSIM-N": coverage.get("ssim"),
        "LPIPS": _mean(pair, "lpips"), "LPIPS-N": coverage.get("lpips"),
        "M-LMD": _mean(pair, "mouth_lmd"), "F-LMD": _mean(pair, "face_lmd"), "LMD-N": coverage.get("lmd"),
        "EmotiEff-Acc": emot.get("accuracy_v4") if isinstance(emot, dict) else None,
        "EmotiEff-N": coverage.get("emotieff"),
        "DFER-CLIP-Acc": dfer.get("accuracy_v4") if isinstance(dfer, dict) else None,
        "DFER-N": coverage.get("dfer"), "Protocol": PROTOCOL_VERSION,
        "Manifest-SHA256": manifest_fingerprint(samples, metrics, context={
            "expected_n": expected_n, "upstream_failures": upstream_for_hash, "device": device_info,
        }),
    }
    report = {
        "protocol_version": PROTOCOL_VERSION, "method": method, "status": status,
        "expected_n": expected_n, "evaluated_n": len(samples), "device": device_info,
        "hard_errors": hard_errors, "errors": hard_errors, "failures": failures,
        "coverage": coverage, "metrics_requested": metrics, "table_row": row,
        "details": details, "per_video": list(per_video.values()),
    }
    return row, report


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns, seen = [], set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key); columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metrics", nargs="+", default=list(PAPER_METRICS), choices=PAPER_METRICS)
    p.add_argument("--force-metrics", nargs="*", default=[], choices=PAPER_METRICS,
                   help="Recompute selected metrics even when a matching completed cache exists.")
    p.add_argument("--device", default="cuda:0",
                   help="Evaluation device. cuda:N selects physical GPU N and isolates all metric subprocesses to it; cpu disables CUDA.")
    p.add_argument("--eval-python", default=_python(DEFAULT_EVAL_PY))
    p.add_argument("--fvd-python", default=_python(DEFAULT_FVD_PY))
    p.add_argument("--lse-python", default=_python(DEFAULT_LSE_PY))
    p.add_argument("--pairwise-python", default=_python(DEFAULT_PAIRWISE_PY) if DEFAULT_PAIRWISE_PY.is_file() else _python(DEFAULT_EVAL_PY))
    p.add_argument("--timeout", type=int, default=7200, help="Per-metric timeout for LSE/FID/FVD/emotion metrics.")
    p.add_argument("--pairwise-timeout", type=int, default=0, help="Pairwise timeout; 0 disables the timeout.")
    p.add_argument("--lse-min-track", type=int, default=5)
    p.add_argument("--fid-frame-stride", type=int, default=1)
    p.add_argument("--fvd-video-length", type=int, default=16)
    p.add_argument("--emotieff-model", default="enet_b2_8")
    p.add_argument("--dfer-segments", type=int, default=16)
    p.add_argument("--expected-n", type=int, default=None)
    p.add_argument("--upstream-failures", default=None)
    p.add_argument("--allow-partial", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest, require_files=True)
    outdir = Path(args.output_dir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); row, report = evaluate(samples, args.method, outdir, list(args.metrics), args)
    report["elapsed_sec"] = time.time() - t0
    metrics_path = outdir / "paper_metrics.json"; table_path = outdir / "paper_table.csv"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(table_path, [row], TABLE_COLUMNS)
    _write_csv(outdir / "per_video.csv", report["per_video"])
    _write_csv(outdir / "failed_samples.csv", report["failures"], FAILURE_COLUMNS)
    trim_paper_table(table_path)
    try:
        report = _load_json(metrics_path); row = report.get("table_row", row)
    except Exception:
        pass
    print(f"[paper-eval] method={args.method} status={row['Status']} N={row['N']} evaluated={row['Evaluated-N']}")
    print(f"[paper-eval] table: {table_path}")
    print(f"[paper-eval] per-video records: {outdir / 'per_video.csv'}")
    if report.get("failures"):
        print(f"[paper-eval] failed samples: {outdir / 'failed_samples.csv'}", file=sys.stderr)
        for f in report["failures"]:
            print(f"  FAIL [{f.get('metric')}] {f.get('name') or f.get('fake') or '<global>'}: {f.get('error')}", file=sys.stderr)
    for error in report.get("hard_errors", []):
        print(f"  ERROR: {error}", file=sys.stderr)
    return 2 if row["Status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
