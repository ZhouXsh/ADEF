#!/usr/bin/env python3
"""Generate fake videos for one baseline without running paper metrics.

This is the generation-only companion to ``final_evaluator.py``. It accepts the
same ``image,audio,gt_video[,emotion]`` triples file, but runs exactly one
baseline selected by ``--method`` and writes outputs under
``RESULT/<method>/videos``. Run one process per baseline to generate the five
methods in parallel.

Existing target videos are reused by the shared ``generate_one`` helper, so a
stopped run can be launched again and continue from already generated samples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from final_evaluator import (  # noqa: E402
    BASELINE_METHODS,
    DEFAULT_RESULT_ROOT,
    _import_baselines,
    _sample_name,
    generate_one,
    read_triples,
)
from paper_protocol import Sample, write_manifest  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--method", required=True, choices=BASELINE_METHODS,
                   help="Generate exactly one baseline so methods can be run in separate processes.")
    p.add_argument("--triples-file", required=True,
                   help="Text file with image,audio,gt_video[,emotion] per line.")
    p.add_argument("--output-root", default=str(DEFAULT_RESULT_ROOT),
                   help="Root directory; videos are written to <output-root>/<method>/videos/.")
    p.add_argument("--scenario", default="paper_eval",
                   help="Scenario string forwarded to the existing baseline runner.")
    p.add_argument("--cuda-visible-devices", default=None,
                   help="Optional CUDA_VISIBLE_DEVICES value set before importing the baseline runner.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate triples and print planned work without generating videos.")
    return p.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _status_payload(method: str, expected_n: int, valid_inputs: int,
                    generated: list[tuple[int, object, Path]], failures: list[dict],
                    attempted: int) -> dict:
    return {
        "method": method,
        "expected": expected_n,
        "valid_inputs": valid_inputs,
        "attempted": attempted,
        "generated": len(generated),
        "pending": max(0, valid_inputs - attempted),
        "failures": failures,
    }


def main() -> int:
    args = parse_args()

    indexed_triples, input_failures, expected_n = read_triples(args)
    root = Path(args.output_root).resolve()
    method_dir = root / args.method
    videos_dir = method_dir / "videos"

    print(
        f"[baseline-gen] method={args.method} samples={expected_n} "
        f"valid_inputs={len(indexed_triples)} output={videos_dir}"
    )
    if args.method == "eat_code":
        print("[baseline-gen] EAT receives GT-derived driving pose, matching final_evaluator.py.")

    if args.dry_run:
        for i, t in indexed_triples:
            print(i, t)
        for failure in input_failures:
            print("INPUT FAIL", failure, file=sys.stderr)
        return 0 if not input_failures else 1

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"[baseline-gen] CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    method_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    rb = _import_baselines()
    if args.method not in rb.METHODS:
        print(f"ERROR: baseline runner does not expose method {args.method!r}", file=sys.stderr)
        return 2

    generated: list[tuple[int, object, Path]] = []
    failures: list[dict] = list(input_failures)
    status_path = method_dir / "generation_status.json"
    upstream_path = method_dir / "upstream_failures.json"

    for attempt_no, (i, t) in enumerate(indexed_triples, start=1):
        try:
            t0 = time.time()
            fake = generate_one(rb, args.method, t, i, videos_dir, args.scenario)
            generated.append((i, t, fake))
            print(
                f"[{args.method}] [{i + 1}/{expected_n}] OK "
                f"{fake.name} ({time.time() - t0:.1f}s)",
                flush=True,
            )
        except KeyboardInterrupt:
            payload = _status_payload(
                args.method, expected_n, len(indexed_triples), generated, failures, attempt_no - 1
            )
            _write_json(status_path, payload)
            _write_json(upstream_path, {"failures": failures})
            print(f"\n[{args.method}] interrupted; progress saved to {status_path}", file=sys.stderr)
            return 130
        except Exception as exc:
            name = _sample_name(t, i)
            failure = {
                "stage": "generation",
                "name": name,
                "fake": "",
                "gt": t.gt,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(
                f"[{args.method}] [{i + 1}/{expected_n}] FAIL {name}: {exc}",
                file=sys.stderr,
                flush=True,
            )

        payload = _status_payload(
            args.method, expected_n, len(indexed_triples), generated, failures, attempt_no
        )
        _write_json(status_path, payload)
        _write_json(upstream_path, {"failures": failures})

    samples = [
        Sample(
            name=_sample_name(t, i),
            fake=str(fake),
            gt=t.gt,
            emotion=t.emotion,
            image=t.image,
            audio=t.audio,
        )
        for i, t, fake in generated
    ]
    if samples:
        manifest = write_manifest(method_dir / "manifest.csv", samples)
        print(f"[{args.method}] manifest: {manifest}")

    final_payload = _status_payload(
        args.method, expected_n, len(indexed_triples), generated, failures, len(indexed_triples)
    )
    _write_json(status_path, final_payload)
    _write_json(upstream_path, {"failures": failures})

    print(
        f"[{args.method}] finished: generated={len(generated)}/{expected_n}, "
        f"failures={len(failures)}"
    )

    if len(generated) == 0:
        return 2
    return 0 if not failures and len(generated) == expected_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
