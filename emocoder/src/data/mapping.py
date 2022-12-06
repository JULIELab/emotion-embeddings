
#from git import Repo
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from urllib.request import urlretrieve
from zipfile import ZipFile
import io
import numpy as np
import os
import torchvision
from PIL import Image
import json
import pytreebank
import re
import tempfile
from emocoder.src.utils import get_split, to_one_hot
from emocoder.src import metrics, constants, utils
from emocoder.src.data import words, images

from .words import ANEW1999, Stevenson2007, XANEW, NRC
from .utils import MinMaxScaler
from typing import Dict, List


class MappingDataset(Dataset):
    """
    Dataset for emotion representation mapping.

    Features and labels are dicts formatname --> numpy array

    Classmethod `load_data` needs to be implemented in every subclass
    """
    format: List[str]  # specific to mapping dataset, the names of the two annotation formats
    variables: Dict[str, list]
    metric: Dict[str, metrics.Metric.__class__]
    loss: Dict[str, torch.nn.Module]
    problem: Dict[str, list]
    scaling: Dict[str, str]
    performance_key: Dict[str, str]
    greater_is_better: Dict[str, bool]




    @classmethod
    def load_data(cls, split):
        """

        :return: tuple (features_key, labels) of dicts, each dict consists stores numpy arrays according to one of the two formats
        """
        raise NotImplementedError
        # features_key, labels = None, None
        # return features_key, labels


    def __init__(self, split):
        """

        :param named_arrays: dict str -> numpy 1D or 2D array (dict is nice because there can only be one array
                            per format
        """

        assert split in ["train", "dev", "test", "full"]
        self.split = split

        for frmt in self.format:
            assert frmt in constants.FORMATS

        # check for data consistency
        for dc in [self.variables, self.metric, self.loss, self.problem]:
            assert sorted(self.format) == sorted(list(dc.keys()))


        # load data and check that the keys match
        self.features, self.labels = self.load_data(self.split)
        assert sorted(list(self.features.keys())) == sorted(self.format)
        assert sorted(list(self.labels.keys())) == sorted(self.format)


        # check that everything has the same lenght
        features_0 = self.features[self.format[0]]
        features_1 = self.features[self.format[1]]
        labels_0 = self.labels[self.format[0]]
        labels_1 = self.labels[self.format[1]]

        for array in [features_0, features_1, labels_0, labels_1]:
            assert len(array) == len(features_0)
            assert isinstance(array, np.ndarray)




    def __len__(self):
        return len(self.features[self.format[0]])

    def __getitem__(self, idx):

        sample = {}
        for key in self.format:
            sample[f"features_{key}"] = self.features[key][idx]
            sample[f"labels_{key}"] = self.labels[key][idx]
            sample["ids"] = idx # just because...

        return sample


    @classmethod
    def score(cls,
              features_key: str,  # string identifier of one of the two label sets
              labels_key: str,  # "
              model,
              device,
              split: str,
              batch_size=512):

        ds = cls(split)
        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)
        model.eval()

        metric = cls.metric[labels_key](vars=cls.variables[labels_key])  # use dataset class
        loss_fn = cls.loss[labels_key]()
        with torch.no_grad():
            val_loss = torch.tensor(0.)
            for batch in dl:
                features = batch[f"features_{features_key}"].to(device)
                labels = batch[f"labels_{labels_key}"].to(device)
                preds = model(features)
                metric.update(true=labels, pred=preds)
                val_loss += loss_fn(preds, labels)
        return metric, val_loss.item()



class ANEW_Stevenson(MappingDataset):

    format = ["vad", "be5"]

    variables = {"vad": constants.VAD,
                 "be5": constants.BE5}

    loss = {"vad": torch.nn.MSELoss,
            "be5": torch.nn.MSELoss}

    metric = {"vad": metrics.Pearson,
              "be5": metrics.Pearson}

    problem = {"vad": constants.MULTIVARIATE_REGRESSION,
               "be5": constants.MULTIVARIATE_REGRESSION}

    scaling = {"vad": constants.TANH,
               "be5": constants.SIGMOID}

    performance_key = {"vad": "mean",
                        "be5": "mean"}
    greater_is_better = {"vad": True,
                        "be5": True}


    @classmethod
    def load_data(cls, split):

        anew = ANEW1999.get_df()
        steve = Stevenson2007.get_df()

        indices = get_split("ANEW-Stevenson")[split]
        anew = anew.loc[indices]
        steve = steve.loc[indices]

        scaler = MinMaxScaler(1, 9, -1, 1)
        anew = anew.applymap(scaler)
        scaler = MinMaxScaler(1, 5, 0, 1)
        steve = steve.applymap(scaler)

        anew = np.asarray(anew, dtype=np.float32)
        steve = np.asarray(steve, dtype=np.float32)
        features = {"vad": anew, "be5": steve}
        labels = {"vad": anew, "be5":steve}

        return features, labels






class ANEW_VA_Stevenson(ANEW_Stevenson):
    """
    Hacky solution to allow testing on VA-part of ANEW as features_key only.
    """
    format = ["va", "be5"]

    variables = {"va": constants.VA,
                 "be5": constants.BE5}

    loss = {"va": torch.nn.MSELoss,
            "be5": torch.nn.MSELoss}

    metric = {"va": metrics.Pearson,
              "be5": metrics.Pearson}

    problem = {"va": constants.MULTIVARIATE_REGRESSION,
               "be5": constants.MULTIVARIATE_REGRESSION}

    scaling = {"va": constants.TANH,
               "be5": constants.SIGMOID}

    performance_key = {"va": "mean",
                       "be5": "mean"}

    greater_is_better = {"va": True,
                         "be5": True}

    @classmethod
    def load_data(cls, split):

        anew = words.ANEW1999_VA.get_df() # only change here compared to above
        steve = Stevenson2007.get_df()

        indices = get_split("ANEW-Stevenson")[split]
        anew = anew.loc[indices]
        steve = steve.loc[indices]

        scaler = MinMaxScaler(1, 9, -1, 1)
        anew = anew.applymap(scaler)
        scaler = MinMaxScaler(1, 5, 0, 1)
        steve = steve.applymap(scaler)

        anew = np.asarray(anew, dtype=np.float32)
        steve = np.asarray(steve, dtype=np.float32)
        features = {"va": anew, "be5": steve}
        labels = {"va": anew, "be5":steve}

        return features, labels







class XANEW_NRC(MappingDataset):

    format = ["vad", "nrc"]

    variables = {"vad": words.XANEW.variables,
                 "nrc": words.NRC.variables}

    loss = {"vad": words.XANEW.loss,
            "nrc": words.NRC.loss}

    metric = {"vad": words.XANEW.metric,
              "nrc": words.NRC.metric}

    problem = {"vad": constants.MULTIVARIATE_REGRESSION,
               "nrc": constants.MULTILABEL}

    scaling = {"vad": constants.TANH,
               "nrc": constants.LOGITS}

    performance_key = {"vad": "mean",
                       "nrc": "f1_mean"}
    greater_is_better = {"vad": True,
                         "nrc": True}


    @classmethod
    def load_data(cls, split):
        nrc = NRC.get_df()
        xanew = XANEW.get_df()

        indices = get_split("XANEW-NRC")[split]
        nrc = nrc.loc[indices]
        xanew = xanew.loc[indices]
        xanew = XANEW.scale(xanew)

        nrc = np.asarray(nrc, dtype=np.float32)
        xanew = np.asarray(xanew, dtype=np.float32)

        features = {"vad": xanew, "nrc": nrc}
        labels = {"vad": xanew, "nrc": nrc}

        return features, labels






class FER_BE_VAD(MappingDataset):

    format = ["vad", "be_fer13"]

    variables = {"vad": images.FER2013Vad.variables,
                 "be_fer13": images.FER2013.variables}

    loss = {"vad": images.FER2013Vad.loss,
            "be_fer13": images.FER2013.loss}

    metric = {"vad": images.FER2013Vad.metric,
              "be_fer13": images.FER2013.metric}

    problem = {"vad": images.FER2013Vad.problem,
               "be_fer13": images.FER2013.problem}

    scaling = {"vad": constants.TANH,
               "be_fer13": constants.LOGITS}

    performance_key = {"vad": "mean",
                       "be_fer13": "acc"}

    greater_is_better = {"vad": True,
                         "be_fer13": True}

    @classmethod
    def load_data(cls, split):

        df = pd.read_csv(images.FER2013Base.path, index_col=0)
        if split == "train":
            df = df[df.Usage == "Training"]
        elif split == "dev":
            df = df[df.Usage == "PublicTest"]
        elif split == "test":
            df = df[df.Usage == "PrivateTest"]
        elif split == "full":
            pass
        else:
            raise ValueError()

        be_labels = images.FER2013.get_ratings(df)
        be_features = to_one_hot(be_labels, len(images.FER2013.variables), dtype=np.float32)

        vad_ratings = images.FER2013Vad.get_ratings(df)

        features = {"be_fer13": be_features, "vad": vad_ratings}
        labels = {"be_fer13": be_labels, "vad": vad_ratings}

        return features, labels


class AffectNet_Mapping(MappingDataset):
    format = ["va", "be_affectnet"]

    variables = {"va": constants.VA,
                 "be_affectnet": constants.BE_AFFECTNET}

    loss = {"va": torch.nn.MSELoss,
            "be_affectnet": torch.nn.CrossEntropyLoss}

    metric = {"va": metrics.Pearson,
              "be_affectnet": metrics.MulticlassAccuracy}

    problem = {"va": constants.MULTIVARIATE_REGRESSION,
               "be_affectnet": constants.MULTICLASS}

    scaling = {"va": constants.TANH,
               "be_affectnet": constants.LOGITS}

    performance_key = {"va": "mean",
                       "be_affectnet": "acc"}
    greater_is_better = {"va": True,
                         "be_affectnet": True}

    dir_path = utils.get_dataset_dir() / "AffectNet" / "Labels" / "ManuallyAnnotated"

    # subDirectory_filePath, face_x, face_y, face_width, face_height, facial_landmarks, expression, valence, arousal
    #0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt, 8: None, 9: Uncertain, 10: No - Face

    @classmethod
    def load_data(cls, split):

        if split in ["dev", "test"]:
            file_path = cls.dir_path / "validation.csv"
            names = ["subDirectory_filePath", "face_x", "face_y", "face_width", "face_height", "facial_landmarks", "expression", "valence", "arousal"]
            df = pd.read_csv(file_path, names=names, usecols=["expression", "valence", "arousal"])
            half_index = int(len(df) / 2)

            if split == "dev": df = df.iloc[:half_index]
            elif split == "test": df = df.iloc[half_index:]
            else: raise ValueError

        elif split=="train":
            file_path = cls.dir_path / "training.csv"
            df = pd.read_csv(file_path, header=0, usecols=["expression", "valence", "arousal"])


        else:
            raise ValueError

        # remove the following classes form the dataset: 8: None, 9: Uncertain, 10: No - Face
        df = df[df.expression < 8]

        be_labels = np.asarray(df.expression)
        be_features = to_one_hot(be_labels.astype(np.int), len(cls.variables["be_affectnet"]), dtype=np.float32)

        
        va_labels = np.asarray(df[["valence", "arousal"]], dtype=np.float32)
        va_features = va_labels

        features = {"be_affectnet": be_features, "va": va_features}
        labels = {"be_affectnet": be_labels, "va": va_labels}

        return features, labels

