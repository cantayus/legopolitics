from __future__ import annotations

import argparse

from legopolitics.review.app import run_app

parser = argparse.ArgumentParser()
parser.add_argument("--validation", required=True)
parser.add_argument("--corrections", required=True)
args = parser.parse_args()
run_app(args.validation, args.corrections)
