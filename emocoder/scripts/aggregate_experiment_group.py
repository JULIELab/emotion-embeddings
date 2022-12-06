import argparse
from pathlib import Path
import json
from emocoder.src.utils import best_result
import pandas as pd
from collections import defaultdict


def sorter_columns(index):
    columnorder = defaultdict(lambda: 100)
    columnorder.update({
        "acc": 0,
        "valence": 1,
        "arousal": 2,
        "dominance": 3,
        "joy": 4,
        "anger": 5,
        "sadness": 6,
        "fear": 7,
        "disgust": 8,
        "surprise": 9,
        "mean": 999
    })
    return pd.Index([columnorder[i] for i in index])


def sorter_index(index):
    return pd.Index( ["-".join(i.split("-")[:-4]) for i in index] )

def timestamp_remover(i):
    return "-".join(i.split("-")[:-4])



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Aggregates results of a group of experiments into a csv file.")
    parser.add_argument("dir", help="The directory of the experiment group. The subdirectories are the "
                                     "directories of the experiments.")
    parser.add_argument("--multitask", action="store_true", default=False, help="Mulitask experiments produce multiindices.")
    parser.add_argument("--removepr", action="store_true", default=False, help="Whether precision and recall values "
                                                                                "should be removed from overview.")
    parser.add_argument("--removetimestamp", action="store_true", default=False, help="Cuts of the timestamp from "
                                                                                      "experiment names.")

    args = parser.parse_args()

    dir = Path(args.dir).resolve()

    experiments = dir.iterdir()
    experiments = list(filter(lambda p: p.is_dir(), experiments))


    if len(experiments) == 0:
        print(f"FAILED aggregation of experiments in {dir}: Nothing to aggregate!")

    else:

        overall = {}

        # get best results from experiement one after the other
        for exp in experiments:
            if exp.is_dir():
                with open(exp/"config.json") as f:
                    config = json.load(f)
                    name = exp.name
                    result = best_result(exp,
                                         performance_key=config["performance_key"],
                                         greater_is_better=config["greater_is_better"])
                    overall[name] = result


        df = pd.DataFrame.from_dict(overall, orient="index")

        # sorting the dataframe,
        df = df.sort_index(axis=1, key=sorter_columns)
        df = df.sort_index(axis=0, key=sorter_index, ascending=False)

        if args.multitask:
            # make new multiindex
            new_index = pd.MultiIndex.from_product([df.index, df.columns])
            #get all possible columns:
            columns = set()
            for i in df.index:
                for j in df.columns:
                    canditate = df.loc[i,j]
                    if isinstance(canditate, dict):
                        columns = columns.union(set(canditate.keys()))
                    else:
                        pass

            df_new = pd.DataFrame(index=new_index, columns=columns).sort_index(axis=1, key=sorter_columns)
            for i in df.index:
                for j in df.columns:
                    canditate = df.loc[i,j]
                    if isinstance(canditate, dict):
                        df_new.loc[(i,j)] = canditate
                    else:
                        pass
            df = df_new

        if args.removepr:
            rename = {}
            drop = []
            for c in df.columns:
                if c.startswith("prec_") or c.startswith("rec_"):
                    drop.append(c)
                elif c.startswith("f1_"):
                    rename[c] = c[3:]
                else:
                    pass
            df = df.drop(columns=drop)
            for old, new in rename.items():
                if new in df.columns:
                    df[new].update(df[old])
                    df = df.drop(columns=old)
                else:
                    df = df.rename(columns={old:new})

        if args.removetimestamp:
            df = df.rename(mapper=timestamp_remover, axis=0, level=0)



        df.to_csv(dir / "overview.csv")
        print(f"Successfully aggregated experiments in {dir}")


