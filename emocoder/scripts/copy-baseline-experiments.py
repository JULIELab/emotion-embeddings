#!/usr/bin/env python

import argparse
from pathlib import Path
import logging
from emocoder.src.utils import get_project_root

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s:%(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A command line script that copies (by symlinking) folders "
                                                 "of baseline experiments between target folders, thereby eliminating "
                                                 "the need to manually copy or re-run baseline experiments.")
    parser.add_argument("from_path", type=str, help="Path to target folder where to copy baseline experiments from.")
    parser.add_argument("to_path", type=str, help="Path to target folder where to copy baseline experiments to.")
    args = parser.parse_args()

    root = Path.cwd()
    source = root / Path(args.from_path)
    target = root / Path(args.to_path )

    logging.info(f"Attempting to symlink baseline experiments from {source} to {target}")

    baseline_exps = ["mapping/baseline",
                     "word/baseline", "word/zeroshotbaseline",
                     "text/baseline", "text/zeroshotbaseline",
                     "image/baseline", "image/zeroshotbaseline"]

    for p in baseline_exps:
        for split in ["dev", "test"]:
            for x in (source/p/split).iterdir():
                if x.is_dir():
                    s = source / p / split / x.stem
                    t = target / p / split / x.stem
                    logging.info(f"Now linking {s} to {t}...")
                    (t).symlink_to(s)


    logging.info("Done.")
