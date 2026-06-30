# coding: utf-8
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from eval.common.io import write_json


def parse_result(path: str, stdout: str):
    p = Path(path)
    text = ""
    if p.exists():
        text += p.read_text(encoding="utf-8", errors="ignore")
    text += "\n" + stdout
    try:
        obj = json.loads(text)
        if "fvd" in obj:
            return obj
    except Exception:
        pass
    fvd = None
    for line in text.splitlines():
        low = line.lower().replace(":", " ").replace("=", " ")
        if "fvd" in low:
            for token in reversed(low.split()):
                try:
                    fvd = float(token)
                    break
                except ValueError:
                    continue
    return {"fvd": fvd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--external_cmd", type=str, required=True, help="template with {real_dir},{gen_dir},{out}")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_out = tmp.name
    cmd = args.external_cmd.format(real_dir=args.real_dir, gen_dir=args.gen_dir, out=tmp_out)
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = parse_result(tmp_out, proc.stdout)
    result.update({
        "real_dir": args.real_dir,
        "gen_dir": args.gen_dir,
        "external_cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    })
    write_json(result, args.out)


if __name__ == "__main__":
    main()
