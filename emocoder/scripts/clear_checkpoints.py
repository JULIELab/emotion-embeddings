#!/usr/bin/env python

from pathlib import Path
from emocoder.src.utils import best_result
import argparse


def remove_all_checkpoints_but_i(checkpoint_dir, i):
    for f in checkpoint_dir.iterdir():
        if not f.name == f"model_{i}.pt":
            f.unlink()

def is_exp_dir(dir):
    for x in ["checkpoints", "log", "results.json"]:
        if not (dir/x).exists():
            return False
    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Deletes all but the best checkpoint")
    parser.add_argument("dir", help="The directory of the experiment to delete checkpoints from.")
    parser.add_argument("--recursive", action="store_true", help="apply to all subdirectories recursively")
    parser.add_argument("--performance_key", help="The key of the performance data in the results.json file.",
                        type=str, default="mean")
    parser.add_argument("--greater_is_better", help="Whether higher values in performance_key correspond to better "
                                                    "performance",
                        type=bool, default=True)
    args = parser.parse_args()

    dir = Path(args.dir).resolve()

    if args.recursive:
        agenda = list(dir.iterdir())
        i = 0
        while i < len(agenda):
            dir = agenda[i]
            #print(agenda)
            #print("\n", dir ,"\n")
            if is_exp_dir(dir):
                try:
                    best = best_result(dir, performance_key=args.performance_key, greater_is_better=args.greater_is_better)
                    epoch = best.name
                    checkpoint_dir = dir / "checkpoints"
                    remove_all_checkpoints_but_i(checkpoint_dir, epoch)
                    print(f"Cleaned up checkpoints in {dir}")
                except KeyError:
                    print(f"KEY ERROR  IN {dir}")
            elif dir.is_dir():
                agenda += list(dir.iterdir())
            i += 1
    else:
        if is_exp_dir(dir):
            best = best_result(dir)
            epoch = best.name
            checkpoint_dir = dir / "checkpoints"
            remove_all_checkpoints_but_i(checkpoint_dir, epoch)
            print(f"Cleaned up checkpoints in {dir}")
        else:
            print("Not an experrimental directory!")
