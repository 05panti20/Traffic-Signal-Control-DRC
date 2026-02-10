 
#!/usr/bin/env python3
"""
Simple, minimal runner: execute the specified .net.xml and .rou.xml with SUMO.

Compatibility: CLI remains the same. Provide either --which (baseline3/baseline6/3/6)
or --file to specify the route file. Use --net to explicitly set the network file.

Behavior simplified: if paths are relative they are resolved relative to the repository root.
If --net is provided it will be used as-is; otherwise the script will try to locate a .net.xml
next to the route file but otherwise will still pass only the route to SUMO.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
import sys


def resolve_repo_path(path: str, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (repo_root / p).resolve()


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Run specified .net.xml and .rou.xml with SUMO")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--route", help="explicit path to a .rou.xml route file")
    parser.add_argument("--gui", action="store_true", help="use sumo-gui instead of sumo")
    parser.add_argument("--net", help="explicit path to a .net.xml file to use instead of auto-detection")
    parser.add_argument("--sumo", help="path to sumo executable (overrides 'sumo'/'sumo-gui' in PATH)")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="extra arguments passed to SUMO")
    # simulation control
    parser.add_argument("--num_seconds", type=int, default=5000, help="number of simulation steps per run (passed to SUMO as --end). Default: 5000")
    parser.add_argument("--episodes", type=int, default=50, help="number of times to repeat the simulation. Default: 50")
    parser.add_argument("--seed-start", type=int, default=None, help="optional starting seed for runs; if provided each run will use seed = seed-start + (i-1). If omitted a per-run seed = run_index is used to vary runs.")

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent

    # determine route file
    if args.route:
        route = resolve_repo_path(args.route, repo_root)
   

    if not route.exists():
        print(f"Route file not found: {route}")
        return 2

    # net file: use explicit --net if given, otherwise try same dir / same stem / fallback
    net = None
    if args.net:
        net_candidate = resolve_repo_path(args.net, repo_root)
        if net_candidate.exists():
            net = net_candidate
        else:
            print(f"Warning: specified net file does not exist: {net_candidate}. Proceeding without -n.")
            net = None
    else:
        # try same directory: route.net.xml or same stem .net.xml or baseline.net.xml
        net_same = route.with_suffix('.net.xml')
        if net_same.exists():
            net = net_same
        else:
            stem_candidate = route.parent / f"{route.stem}.net.xml"
            if stem_candidate.exists():
                net = stem_candidate
            else:
                fallback = route.parent / 'baseline.net.xml'
                if fallback.exists():
                    net = fallback

    # choose sumo binary
    if args.sumo:
        sumo_bin = Path(args.sumo)
    else:
        sumo_bin_name = 'sumo-gui' if args.gui else 'sumo'
        sumo_path = shutil.which(sumo_bin_name)
        sumo_bin = Path(sumo_path) if sumo_path else Path(sumo_bin_name)

    base_cmd = [str(sumo_bin)]
    if net is not None:
        base_cmd += ['-n', str(net)]
    base_cmd += ['-r', str(route)]
    if args.extra:
        base_cmd += args.extra

    # run repeatedly
    last_rc = 0
    for i in range(1, args.episodes + 1):
        run_cmd = list(base_cmd)
        # pass end/steps
        if args.num_seconds is not None:
            run_cmd += ['--end', str(args.num_seconds)]
        # set seed per run to vary runs
        if args.seed_start is not None:
            seed_val = args.seed_start + (i - 1)
            run_cmd += ['--seed', str(seed_val)]
        else:
            # default: use run index as seed to vary runs
            run_cmd += ['--seed', str(i)]

        print(f'Run {i}/{args.episodes} — command:')
        print(' '.join(run_cmd))

        try:
            completed = subprocess.run(run_cmd)
            last_rc = completed.returncode
            if last_rc != 0:
                print(f"Run {i} exited with code {last_rc}; stopping early.")
                break
        except FileNotFoundError:
            print(f"SUMO executable not found: {sumo_bin}. Make sure SUMO is installed and on PATH, or pass --sumo path.")
            return 3

    return last_rc


if __name__ == '__main__':
    raise SystemExit(main())
