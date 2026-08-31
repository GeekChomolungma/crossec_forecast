#!/usr/bin/env python
"""Expand a sweep spec into N fully-resolved configs and dispatch them.

    python scripts/sweep.py -s experiments/sweep.example.yaml --launcher print
    python scripts/sweep.py -s experiments/sweep.example.yaml --launcher local --max-parallel 2
    python scripts/sweep.py -s experiments/sweep.example.yaml --launcher slurm [--dry-run]

One job = one `variant` x one point of the `grid` cartesian product. Each job's config
is materialised to  runs/_sweeps/<stamp>_<spec>/configs/<job>.yaml  so it is completely
self-describing (reproducible, resubmittable, diffable).
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _common import PROJECT_ROOT, bootstrap_path

bootstrap_path()

from omegaconf import OmegaConf  # noqa: E402

from crossec_forecast.pipelines.context import load_config, resolve_run_dir  # noqa: E402

SLURM_TMPL = PROJECT_ROOT / "scripts" / "slurm" / "train.sbatch.tmpl"


# --------------------------------------------------------------------------------------
def build_jobs(spec) -> list[dict]:
    """Return a list of {name, overrides} dicts from the sweep spec."""
    common = list(spec.get("common_overrides", []) or [])
    variants = spec.get("variants", []) or [{"name": "default", "overrides": []}]
    grid = OmegaConf.to_container(spec.get("grid", {}) or {}, resolve=True)

    grid_keys = list(grid.keys())
    grid_points = list(itertools.product(*[grid[k] for k in grid_keys])) or [()]

    jobs: list[dict] = []
    for v in variants:
        v_name = str(v["name"])
        v_over = list(v.get("overrides", []) or [])
        for point in grid_points:
            g_over = [f"{k}={val}" for k, val in zip(grid_keys, point)]
            suffix = "".join(f"__{k.split('.')[-1]}-{val}" for k, val in zip(grid_keys, point))
            jobs.append({"name": f"{v_name}{suffix}", "overrides": common + v_over + g_over})
    return jobs


def materialise(spec, jobs, sweep_root: Path) -> list[dict]:
    """Resolve + save each job's config; attach config_path / run_dir."""
    base_config = str(spec["base_config"])
    cfg_dir = sweep_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    out = []
    for job in jobs:
        overrides = list(job["overrides"]) + [f"run.name={job['name']}"]
        cfg = load_config(base_config, overrides)
        run_dir = resolve_run_dir(cfg, create=False)
        cfg_path = cfg_dir / f"{job['name']}.yaml"
        OmegaConf.save(cfg, cfg_path, resolve=True)
        out.append({**job, "config_path": str(cfg_path), "run_dir": str(run_dir)})
    return out


# --------------------------------------------------------------------------------------
def launch_print(jobs, entry: str) -> None:
    for j in jobs:
        print(f"python scripts/{entry}.py -c {j['config_path']}")


def launch_local(jobs, entry: str, max_parallel: int, sweep_root: Path) -> int:
    log_dir = sweep_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = str(PROJECT_ROOT / "scripts" / f"{entry}.py")

    def _run(job) -> tuple[str, int]:
        log_file = log_dir / f"{job['name']}.log"
        with open(log_file, "w", encoding="utf-8") as fh:
            proc = subprocess.run(
                [sys.executable, script, "-c", job["config_path"]],
                stdout=fh,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
        print(f"[{'ok ' if proc.returncode == 0 else 'FAIL'}] {job['name']}  -> {log_file}")
        return job["name"], proc.returncode

    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        results = list(pool.map(_run, jobs))

    failed = [n for n, rc in results if rc != 0]
    print(f"\n{len(results) - len(failed)}/{len(results)} jobs succeeded.")
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


def launch_slurm(spec, jobs, entry: str, sweep_root: Path, dry_run: bool) -> int:
    """Render scripts/slurm/train.sbatch.tmpl per job into the sweep dir, then sbatch it.

    The template keeps its shell body verbatim (bash ``${...}`` and all); only the
    ``@@TOKEN@@`` markers are substituted here.
    """
    if not SLURM_TMPL.exists():
        raise SystemExit(f"Missing SLURM template: {SLURM_TMPL}")
    tmpl = SLURM_TMPL.read_text(encoding="utf-8")
    s = OmegaConf.to_container(spec.get("slurm", {}) or {}, resolve=True) or {}
    extra = "\n".join(f"#SBATCH {x}" for x in (s.get("extra_sbatch") or []))

    sbatch_dir = sweep_root / "sbatch"
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "@@PROJECT_DIR@@": str(PROJECT_ROOT),
        "@@ENTRY_SCRIPT@@": f"scripts/{entry}.py",
        "@@EXTRA_SBATCH@@": extra,
    }
    for job in jobs:
        rendered = tmpl
        for token, value in {
            **common,
            "@@JOB_NAME@@": job["name"],
            "@@CONFIG_PATH@@": job["config_path"],
        }.items():
            rendered = rendered.replace(token, value)

        sbatch_file = sbatch_dir / f"{job['name']}.sbatch"
        sbatch_file.write_text(rendered, encoding="utf-8")
        if dry_run:
            print(f"[dry-run] wrote {sbatch_file}")
        else:
            subprocess.run(["sbatch", str(sbatch_file)], check=True)
            print(f"[submitted] {job['name']}")
    return 0


# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-s", "--sweep", required=True, help="Path to the sweep spec YAML")
    p.add_argument("--launcher", choices=["local", "slurm", "print"], default=None)
    p.add_argument("--entry", choices=["train", "benchmark"], default=None)
    p.add_argument("--max-parallel", type=int, default=1, help="local launcher concurrency")
    p.add_argument("--dry-run", action="store_true", help="slurm: write .sbatch files but do not submit")
    args = p.parse_args()

    spec = OmegaConf.load(args.sweep)
    launcher = args.launcher or str(spec.get("launcher", "print"))
    entry = args.entry or str(spec.get("entry", "train"))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    sweep_root = PROJECT_ROOT / "runs" / "_sweeps" / f"{stamp}_{Path(args.sweep).stem}"

    jobs = materialise(spec, build_jobs(spec), sweep_root)
    print(f"Sweep '{args.sweep}': {len(jobs)} job(s) | launcher={launcher} | entry={entry}")
    print(f"Artifacts: {sweep_root}\n")

    if launcher == "print":
        launch_print(jobs, entry)
        rc = 0
    elif launcher == "local":
        rc = launch_local(jobs, entry, args.max_parallel, sweep_root)
    else:
        rc = launch_slurm(spec, jobs, entry, sweep_root, args.dry_run)

    sys.exit(rc)


if __name__ == "__main__":
    main()
