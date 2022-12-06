import json
from pathlib import Path
import torch
import time
import logging
import pandas as pd
import numpy as np
import collections.abc


# https://stackoverflow.com/a/3233356/4474892 CC BY-SA
def nested_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_split(name):
    target = get_data_dir() / "splits" / f"{name}_splits.json"
    with open(target) as f:
        dc = json.load(f)
    return dc

def get_project_root():
    return Path(__file__).parents[2].resolve()

def get_target_dir():
    return get_project_root() / "emocoder" / "target"

def get_data_dir():
    return get_project_root() / "emocoder" / "data"

def get_analysis_dir():
    return get_project_root() / "emocoder" / "analysis"

def get_dataset_dir():
    return get_data_dir() / "datasets"

def get_script_dir():
    return get_project_root() / "emocoder" / "scripts"

def get_vector_dir():
    return get_data_dir() / "vectors"

def timestamp():
    return time.strftime("%Y-%m-%d-%H%M%S", time.localtime())

def reset_logger():
    """
    https://stackoverflow.com/a/12158233/4474892
    :return:
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)



def compare_state_dicts(sd1, sd2):
    """
    Compares two PyTorch state dicts. Returns a dictionary of list, containing the names of matching or nonmatching
    parameters, as well as those which are not present in both state dicts
    :param sd1: First state_dict
    :param sd2: Second state_dict
    :return: Dict of Lists of parameters names
    """
    shared_keys = set(sd1.keys()).intersection(set(sd2.keys()))
    only_sd1 = sorted(list(set(sd1.keys()).difference(shared_keys).union()))
    only_sd2 = sorted(list(set(sd2.keys()).difference(shared_keys)))

    shared_keys = sorted(list(shared_keys))
    matching = []
    not_matching = []
    for k in shared_keys:
        if torch.equal(sd1[k], sd2[k]):
            matching.append(k)
        else:
            not_matching.append(k)


    return {"matching": matching,
            "not matching": not_matching,
            "only sd1": only_sd1,
            "only sd2": only_sd2
            }


def best_result(exp_path, performance_key="mean", greater_is_better=True):

    file_path = Path(exp_path / "results.json")
    if file_path.is_file():
        data = pd.read_json(file_path, orient="index")
    else:
        raise FileNotFoundError(f"{file_path} is does not exist or is not a file.")
    best = data.sort_values(performance_key, ascending=(not greater_is_better)).iloc[0]
    return best


def boolean_triangle(matrix):
    mask = np.zeros_like(matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    return mask

def to_one_hot(a, num_classes, dtype=np.float64):
    return np.eye(num_classes, dtype=dtype)[a]

class CELoss_OneHot(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, pred, true):
        _, labels = true.max(dim=1)
        return self._loss(pred, labels)

def cosine(a,b):
    return  np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def get_experiment_dir(base_dir:Path, name_stem:str) -> Path:
    """
     Helper method that returns the correct path of an experiment within  a base directory given only a stem of the
     experiment name, e.g., without timestamp.
     :param base_dir:
     :param name_stem:
     :return: experiment path
     """
    pl = list(base_dir.glob(f"{name_stem}*"))
    assert len(pl) == 1, f"The provided name stem was not unique in base dir {base_dir.resolve()}. Stem: {name_stem}; Candidates: {[p.name for p in pl]}"
    exp = pl[0]
    return exp


class Picking_Layer(torch.nn.Module):

    def __init__(self, pick):
        super().__init__()
        self.pick = pick

    def forward(self, x):
        return torch.unsqueeze(x[:, self.pick], dim=-1)