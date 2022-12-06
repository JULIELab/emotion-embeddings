import abc

import pandas as pd
import torch
import logging
import json
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Union

from emocoder.src import utils, metrics, models


class Experiment:
    """
    This class the use for storing managing experimental data (checkpoints, performance metrics, tensorboard writers,
    outputs,...).
    """

    def __init__(self,
                 name: str,
                 parent_dir,
                 performance_key=None,    # can be set explicitly at init or later when first metric result retrieved
                 greater_is_better=None): # see above

        self.base_name = name
        self.parent_dir = parent_dir
        self.performance_key = performance_key
        self.greater_is_better = greater_is_better


    def order_performance_according_to(self, metric: metrics.Metric):

        if not (self.performance_key is None and self.greater_is_better is None):
            raise ValueError("Order criterions for performance have already been set!")

        self.greater_is_better = metric.greater_is_better
        self.performance_key = metric.performance_key

    @property
    def performance_order_defined(self):
        if self.performance_key is not None and self.greater_is_better is not None:
            return True
        elif self.performance_key is None and self.greater_is_better is None:
            return False
        else:
            raise ValueError("Inconsistent state, order criteria partially set!")


    def  setup(self):
        """
        Call this method first thing in your custom run method!
        This will will create the folder where to put the data.
        """

        if not self.performance_order_defined:
            raise ValueError("Performance order must be defined at this point!")

        #self.current_name = utils.timestamp() + "-" + self.base_name
        self.current_name = self.base_name + "-" + utils.timestamp()

        self.dir = Path(self.parent_dir) /self.current_name
        assert not self.dir.is_dir(), f"Designated data path {self.dir} is already in use!"
        self.dir.mkdir()

        utils.reset_logger() #allowing the logger to write into another file
        logging.basicConfig(level=logging.INFO,
                            format='%(levelname)s:%(asctime)s:%(message)s',
                            handlers=[logging.FileHandler(self.dir / "log"),
                                      logging.StreamHandler()])
        # https: // docs.python.org / 3 / howto / logging.html
        # https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout
        logging.info(f"Starting experiment {self.current_name} of class {type(self)}")
        logging.info(f"Experimental data will be written to {self.dir}")

        logging.info(f"Configuartion: {self.__dict__}")
        with open(self.dir / "config.json", "w") as f:
            json.dump(self.__dict__, fp=f, default=lambda x: str(x))

        # setting up tensorboard
        self.tensorboard_dir = self.dir / "tensorbord"
        self.tensorboard = SummaryWriter(log_dir=self.dir / "tensorbord")
        logging.info(f"Setting up Tensorbord Writer to {self.tensorboard_dir}")

        # setting up checkpoint dir
        self.checkpoint_dir = self.dir / "checkpoints"
        self.checkpoint_dir.mkdir()
        logging.info(f"Checkpoints will be saved to {self.checkpoint_dir}")

        # setting up results.json
        self.results_file = self.dir / "results.json"
        logging.info(f"Results will be written to {self.results_file}")
        self.results = {}


    def save_checkpoint(self, model: torch.nn.Module, name: str):
        checkpoint_name = f"{name}.pt"
        checkpoint_path = self.checkpoint_dir/ checkpoint_name
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Saved model checkpoint to {checkpoint_name}")


    def save_tensorboard_scalars(self, tag:str, value:Union[float, dict], global_step:int):
        if isinstance(value, float):
            self.tensorboard.add_scalar(tag, value, global_step)
        elif isinstance(value, dict):
            # dictionaries might be nested which does not work with tensorboard, need to flatten it first
            # works only for one nesting level!
            dc = value
            rt = {}
            for key, value in dc.items():
                if isinstance(value, dict):
                    for key1, value1 in value.items():
                        rt[f"{key}_{key1}"] = value1
                else:
                    rt[key] = value


            self.tensorboard.add_scalars(tag, rt, global_step)
        else:
            raise ValueError(f"Need to pass float or dict to save_tensorboard_scalars, not {type(value)}")

        self.tensorboard.flush()
        logging.info("Saved results to tensorboard")

    def save_results(self, key, value):
        self.results[key] = value
        with open(self.results_file, "w") as f:
            json.dump(self.results, f)
        logging.info(f"Saved current results")

    def end_experiment(self):
        logging.info(f"End of experiment {self.current_name}.")

    def get_best_result(self):

        assert self.greater_is_better is not None
        assert self.performance_key is not None

        ascending = not self.greater_is_better # if greater is better, than sort descending
        key = self.performance_key
        df = pd.read_json(self.results_file, orient="index")
        df = df.sort_values(by=key, ascending=ascending)
        return df.index[0], df.iloc[0].loc[key] #epoch, result

    def remove_all_checkpoints_but_i(self, i):
        keep_name = f"model_{i}.pt"
        logging.info(f"Removing all checkpoints but {keep_name}...")
        for f in self.checkpoint_dir.iterdir():
            if not f.name == keep_name:
                f.unlink()
                logging.info(f"Checkpoint {f.name} removed.")

    def remove_all_but_best_checkpoint(self):
        epoch, result = self.get_best_result()
        self.remove_all_checkpoints_but_i(epoch)


    def run(self, gpu):
        raise NotImplementedError

    @staticmethod
    def compare_model_with_state_dict(model, state_dict):
        rt = utils.compare_state_dicts(state_dict, model.to(device="cpu").state_dict())
        logging.info(f"Checking that decoders have not been updated: "
                     f"{rt}")


# TODO Started to harmonize experiment classes across modalities, but its just too complicated for now...
# class ProposedExperiment(Experiment):
#
#     def __init__(self,
#                  name:str,
#                  parent_dir: Path,
#                  dataset_class,
#                  epochs,
#                  train_batchsize,
#                  test_batchsize,
#                  ):
#         super().__init__(name=name,
#                          parent_dir=parent_dir,
#                          performance_key=dataset_class.performance_key,
#                          greater_is_better=dataset_class.greater_is_better)
#
#         self.dataset_class = dataset_class
#         self.epochs = epochs
#         self.train_batchsize = train_batchsize
#         self.test_batchsize = test_batchsize
#
#         self.model:  models.EmotionCodec
#
#     @abc.abstractmethod
#     def get_model():
#         raise NotImplementedError
#
#     def run(self, gpu):
#         self.setup()
#         logging.info(f"Starting experiment type {self.__class__} on dataset {self.dataset_class}.")
#         logging.info("Folder structure set up.")
#         logging.info("Building model, loading parameters...")
#
#         is isinstance(self, )
#








def get_best_checkpoint(exp_path: Path):
    with open(exp_path/"config.json") as f:
        config = json.load(f)
    performance_key = config["performance_key"]
    greater_is_better = config["greater_is_better"]
    sort_ascending = not greater_is_better # if greater is better results should be sorted in descending order so ascending should be false
    df = pd.read_json(exp_path / "results.json", orient="index")
    best = df.sort_values(performance_key, ascending=sort_ascending).index[0]
    return exp_path / "checkpoints"/ f"model_{best}.pt"




