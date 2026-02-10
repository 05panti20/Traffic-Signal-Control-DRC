#!/usr/bin/env python3
"""
randomize_vehsperhour.py

Update vehsPerHour attributes in a SUMO .rou.xml file to random integers.

Usage examples:
  python scripts\\randomize_vehsperhour.py --min 200 --max 800
  python scripts\\randomize_vehsperhour.py --infile path\\to\\baseline6.rou.xml --outfile out.rou.xml --seed 42

The script defaults to the repository's baseline6.rou.xml when run from the project root.
"""
import argparse
import os
import random
import sys
import xml.etree.ElementTree as ET


def default_input_path():
    # script is in <repo>/scripts; default routes file is in sumo_rl/nets/baseline
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return os.path.join(repo_root, "sumo_rl", "nets", "baseline", "baseline6.rou.xml")


def parse_args():
    p = argparse.ArgumentParser(description="Randomize vehsPerHour in a SUMO .rou.xml file")
    p.add_argument("--infile", default=default_input_path(), help="Input .rou.xml file")
    p.add_argument("--outfile", default=None, help="Output .rou.xml file (defaults to infile with _rand appended)")
    p.add_argument("--min", type=int, default=300, help="Minimum vehsPerHour (inclusive)")
    p.add_argument("--max", type=int, default=720, help="Maximum vehsPerHour (inclusive)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--per-flow", action="store_true", help="Randomize each flow independently (default true). If not set, one random value is used for all flows")
    return p.parse_args()


def randomize_file(infile, outfile, min_v, max_v, seed=None, per_flow=True):
    if seed is not None:
        random.seed(seed)

    tree = ET.parse(infile)
    root = tree.getroot()

    # Find all flow elements (namespace-agnostic)
    flows = [elem for elem in root.findall('flow')]
    if not flows:
        # try namespaced tags
        flows = [elem for elem in root.findall('.//{*}flow')]

    if not flows:
        print("No <flow> elements found in", infile, file=sys.stderr)
        return 1

    if not per_flow:
        val = str(random.randint(min_v, max_v))
        for f in flows:
            f.set('vehsPerHour', val)
        print(f"Set vehsPerHour={val} for all {len(flows)} flows")
    else:
        for f in flows:
            val = str(random.randint(min_v, max_v))
            f.set('vehsPerHour', val)
        print(f"Randomized vehsPerHour for {len(flows)} flows (range {min_v}-{max_v})")

    # Determine output path
    if outfile is None:
        base, ext = os.path.splitext(infile)
        outfile = base + "_rand" + ext

    # Write back the XML. Use encoding UTF-8.
    tree.write(outfile, encoding='utf-8', xml_declaration=True)
    print("Wrote randomized routes to:", outfile)
    return 0


def main():
    args = parse_args()

    if not os.path.isfile(args.infile):
        print(f"Input file not found: {args.infile}", file=sys.stderr)
        return 2

    if args.min > args.max:
        print("--min must be <= --max", file=sys.stderr)
        return 3

    return randomize_file(args.infile, args.outfile, args.min, args.max, seed=args.seed, per_flow=args.per_flow)


if __name__ == '__main__':
    sys.exit(main())
