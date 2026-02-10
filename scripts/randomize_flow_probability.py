#!/usr/bin/env python3
"""Randomize <flow>@probability in a SUMO .rou.xml file.

Simple utility: exposes randomize_flow_probability(...) and a small CLI.
"""

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def strip_ns(tag: str) -> str:
    return tag.split('}', 1)[-1] if '}' in tag else tag


def randomize_flow_probability(
    input_path: str,
    output_path: Optional[str] = None,
    min_prob: float = 0.0,
    max_prob: float = 1.0,
    seed: Optional[int] = None,
    precision: int = 4,
):
    """Randomize probability for all <flow> elements in input_path XML.

    If output_path is None or equals input_path, the input file is overwritten.
    Returns the number of flows updated.
    """
    inp = Path(input_path)
    if output_path is None:
        out = inp
    else:
        out = Path(output_path)

    if not inp.exists():
        raise FileNotFoundError(f"input file not found: {inp}")

    if min_prob < 0 or max_prob < 0 or max_prob < min_prob:
        raise ValueError("invalid min/max range")

    if seed is not None:
        random.seed(seed)

    tree = ET.parse(str(inp))
    root = tree.getroot()

    flows = [elem for elem in root.iter() if strip_ns(elem.tag) == "flow"]

    for f in flows:
        val = random.uniform(min_prob, max_prob)
        prob_str = f"{val:.{precision}f}"
        f.set("probability", prob_str)

    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out), encoding="utf-8", xml_declaration=True)

    return len(flows)


def _parse_args():
    p = argparse.ArgumentParser(description="Randomize <flow>@probability in a SUMO .rou.xml file")
    p.add_argument("-i", "--input", required=True, help="Input .rou.xml path")
    p.add_argument("-o", "--output", help="Output .rou.xml path (if omitted, overwrites input)")
    p.add_argument("--min", type=float, default=0.0, help="Minimum probability (inclusive), default 0.0")
    p.add_argument("--max", type=float, default=1.0, help="Maximum probability (inclusive), default 1.0")
    p.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    p.add_argument("--precision", type=int, default=4, help="Number of decimal places to write (default 4)")
    return p.parse_args()


def main():
    args = _parse_args()
    n = randomize_flow_probability(
        args.input, args.output, args.min, args.max, args.seed, args.precision
    )
    print(f"Wrote {n} randomized flow probability values to: {args.output or args.input}")


if __name__ == "__main__":
    main()
