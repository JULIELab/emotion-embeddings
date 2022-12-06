#from git import Repo
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
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
from torch.utils.data import DataLoader
from emocoder.src.utils import get_split
from emocoder.src import constants, utils

from .utils import MinMaxScaler, BaseDataset
from .. import metrics
from typing import Type

from unicode_tr import unicode_tr


class WordDataset(BaseDataset):


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {'raw': self.df.index[idx],
                  'labels': self.df[self.variables].iloc[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


    @classmethod
    def score(cls,
              model: torch.nn.Module,
              device: torch.device,
              split: str,
              transform: callable,
              batch_size: int = 512, # changing default batch size compare to superclass
              collater: callable = None,
              metric: Type[metrics.Metric] = None):

        return super().score(model, device, split, transform, batch_size, collater, metric)




class MultivariateRegressionDataset(WordDataset):
    loss = torch.nn.MSELoss
    metric = metrics.Pearson
    performance_key = "mean"
    greater_is_better = True
    problem = constants.MULTIVARIATE_REGRESSION


class MultiLabelClassificationDataset(WordDataset):
    loss = torch.nn.BCEWithLogitsLoss
    metric = metrics.MultiLabelF1
    performance_key = "f1_mean"
    greater_is_better = True
    problem = constants.MULTILABEL


class XANEW(MultivariateRegressionDataset):

    format = "vad"
    variables = constants.VAD
    scaling = "tanh"

    @classmethod
    def get_df(cls, path=utils.get_dataset_dir() / 'XANEW.zip',
               url="https://github.com/JULIELab/XANEW/archive/master.zip"):

        if not path.is_file():
           print('Downloading data ...')
           urlretrieve(url, path)

        with ZipFile(path) as zip:
            with zip.open('XANEW-master/Ratings_Warriner_et_al.csv') as csv:
               df = pd.read_csv(csv, index_col=0, keep_default_na=False, sep=',', usecols=[0, 1, 2, 5, 8])
               df = df[['Word', 'V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']]
               df.columns = ['word', 'valence', 'arousal', 'dominance']
               df.set_index('word', inplace=True)

        return df

    @staticmethod
    def scale(df):
        return df.applymap(MinMaxScaler(1, 9, -1, 1))

    def __init__(self,
             split='full',
             transform=None,
             #vars=['valence', 'arousal', 'dominance'], #exclude variables if needed
             path=Path.home() / 'data' / 'XANEW.zip',
             url="https://github.com/JULIELab/XANEW/archive/master.zip",
             scale=True):


        self.df = self.get_df(path, url)
        with ZipFile(path) as zip:
            if not split=='full':
                with zip.open('XANEW-master/splits/crosslingual/{}.txt'.format(split)) as f:
                    words = f.read().decode("utf-8").split('\n')
                    self.df = self.df.loc[words]

        if scale:
            self.df = self.scale(self.df)

        super(XANEW, self).__init__(split, transform)

# class XANEW_BE(MultivariateRegressionDataset):
#
#     format = "be5"
#     variables = constants.BE5
#     scaling = "sigmoid"
#
#     @staticmethod
#     def _get_df(clip=True): # this gets the dataset created by representation mapping
#
#         # have to download xanew as well to get the splits
#         url = "https://github.com/JULIELab/XANEW/archive/master.zip"
#         path = utils.get_dataset_dir() / 'XANEW.zip'
#         if not path.is_file():
#             print("Downloading data ... ")
#             urlretrieve(url, path)
#
#         url = "https://github.com/JULIELab/EmoMap/archive/master.zip"
#         path = Path.home() / 'data' / "EmoMap.zip"
#         if not path.is_file():
#             print("Downloading data ... ")
#             urlretrieve(url, path)
#
#         with ZipFile(path) as zip:
#             with zip.open("EmoMap-master/coling18/main/lexicon_creation/lexicons/Warriner_BE.tsv") as f:
#                 df = pd.read_csv(f, sep="\t", index_col=0)
#
#         if clip: # because train and dev sets are predictions, some of them are outside the standard range
#             df = df.clip(lower=1, upper=5)
#
#         # convert col names to lowercase
#         df = df.rename({c:c.lower() for c in df.columns}, axis=1)
#
#         return df
#
#
#     def __init__(self, split, transform=None, scale=True):
#         assert split in ["train", "dev", "test"]
#
#         self.df = self._get_df()
#         if split == "test": # return stevenson dataset
#             path = utils.get_dataset_dir() / "stevenson2007.xls"
#             self.df = self.df[['word', 'mean_hap', 'mean_ang', 'mean_sad',
#                      'mean_fear', 'mean_dis']]
#             self.df.columns = ['word', 'joy', 'anger', 'sadness', 'fear', 'disgust']
#             self.df.set_index('word', inplace=True)
#         else:
#             self.df = self._get_df()
#             # getting xanew datasplit
#             path = Path.home() / 'data' / 'XANEW.zip'
#             with ZipFile(path) as zip:
#                 with zip.open('XANEW-master/splits/crosslingual/{}.txt'.format(split)) as f:
#                     words = f.read().decode("utf-8").split('\n')
#             words = list(set(words).intersection(set(self.df.index)))
#             self.df = self.df.loc[words]
#
#         if scale:
#             self.df = self.df.applymap(MinMaxScaler(1, 5, 0, 1))
#
#         super().__init__(split, transform)

class Stadthagen_VA(MultivariateRegressionDataset):

    format = "va"
    variables = constants.VA
    scaling = "tanh"

    @classmethod
    def get_df(cls):
        path = utils.get_dataset_dir() / "Stadthagen_VA.csv"
        df = pd.read_csv(path, encoding='cp1252')
        df = df[['Word', 'ValenceMean', 'ArousalMean']]
        df.columns = ['word', 'valence', 'arousal']
        df.word = [w[:-1] if w.endswith("*") else w for w in df.word]
        df.set_index('word', inplace=True)
        return df

    def __init__(self, split, transform=None, scale=True):
        assert split in ["train", "dev", "test", "full"]
        self.df = self.get_df()
        if split != "full":
            indices = get_split("StadthagenVA")[split]
            self.df = self.df.loc[indices]
        if scale:
            self.df = self.df.applymap(MinMaxScaler(1, 9, -1, 1))
        super().__init__(split, transform)


class Stadthagen_BE(MultivariateRegressionDataset):

    format = "be5"
    variables = constants.BE5
    scaling = "sigmoid"

    @classmethod
    def get_df(cls):
        path = utils.get_dataset_dir() / "Stadthagen_BE.csv"
        df = pd.read_csv(path, encoding='cp1252')
        df = df[['Word', 'Happiness_Mean', 'Anger_Mean', 'Sadness_Mean', 'Fear_Mean', 'Disgust_Mean']]
        df.columns = ['word', 'joy', 'anger', 'sadness', 'fear', 'disgust']
        df.set_index('word', inplace=True)
        return df

    def __init__(self, split, transform=None, scale=True):
        assert split in ["train", "dev", "test", "full"]
        self.df = self.get_df()
        if split != "full":
            indices = get_split("StadthagenBE")[split]
            self.df = self.df.loc[indices]
        if scale:
            self.df = self.df.applymap(MinMaxScaler(1, 5, 0, 1))
        super().__init__(split, transform)


class Imbir(MultivariateRegressionDataset):

    format = "vad"
    variables = constants.VAD
    scaling = "tanh"

    @classmethod
    def get_df(cls):
        path = utils.get_dataset_dir() / "Imbir.xlsx"
        pl_gold = pd.read_excel(path,
                                index_col=0)
        rename = {'Valence_M': 'valence',
                  'arousal_M': 'arousal',
                  'dominance_M': 'dominance',
                  'polish word': 'word',
                  }
        pl_gold.rename(columns=rename,
                       inplace=True)
        pl_gold.set_index('word', inplace=True)

        for c in pl_gold.columns:
            if c not in rename.values():
                pl_gold.drop(columns=c, inplace=True)


        # Removing multi-token entries
        def to_keep(s):
            if " " in s: return False
            else: return True

        pl_gold = pl_gold[[to_keep(w) for w in pl_gold.index]]

        return pl_gold

    def __init__(self, split, transform=None, scale=True):
        assert split in ["train", "dev", "test"]
        indices = get_split("Imbir")[split]
        self.df = self.get_df().loc[indices]
        if scale:
            self.df = self.df.applymap(MinMaxScaler(1, 9, -1, 1))
        super().__init__(split, transform)

class Riegel(MultivariateRegressionDataset):

    format = "va"
    variables = constants.VA
    path = utils.get_dataset_dir() / "Riegel2015.xlsx"
    split_key = "Riegel"
    scaling = constants.TANH

    @classmethod
    def get_df(cls):
        df = pd.read_excel(cls.path, index_col=2)
        df.index.rename('word', inplace=True)
        dc = {
            'val_M_all': 'valence',
            'aro_M_all': 'arousal'
        }
        df.rename(columns=dc, inplace=True)
        for c in df.columns:
            if c not in dc.values():
                df.drop(columns=c, inplace=True)

        return df

    @staticmethod
    def _scale(df):
        val_scaler = MinMaxScaler(-3, 3, -1, 1)
        aro_scaler = MinMaxScaler(1,5, -1, 1)
        df["valence"] = df["valence"].apply(val_scaler)
        df["arousal"] = df["arousal"].apply(aro_scaler)
        return df


    def __init__(self, split, transform=None, scale=True):
        """
        Boilerplate! This should be inherited somehow!
        :param split:
        :param transform:
        :param scale:
        """
        assert split in ["train", "dev", "test", "full"]
        self.df = self.get_df()
        if split in ["train", "dev", "test"]:
            indices = get_split(self.split_key)[split]
            self.df = self.df.loc[indices]
        if scale:
            self.df = self._scale(self.df)
        super().__init__(split, transform)


class Wierzba(MultivariateRegressionDataset):

    format = "be5"
    variables = constants.BE5
    path = utils.get_dataset_dir() / "Wierzba2015.xlsx"
    split_key = "Wierzba"
    scaling = constants.SIGMOID

    @classmethod
    def get_df(cls):
        df = pd.read_excel(cls.path, index_col=2)
        df.index.rename('word', inplace=True)
        dc = {
            'hap_M_all': 'joy',
            'ang_M_all': 'anger',
            'sad_M_all': 'sadness',
            'fea_M_all': 'fear',
            'dis_M_all': 'disgust'
        }
        df.rename(columns=dc, inplace=True)
        for c in df.columns:
            if c not in dc.values():
                df.drop(columns=c, inplace=True)
        return df

    @staticmethod
    def _scale(df):
        scaler = MinMaxScaler(1, 7, 0, 1)
        df = df.applymap(scaler)
        return df

    def __init__(self, split, transform=None, scale=True):
        """
        Boilerplate! This should be inherited somehow!
        :param split:
        :param transform:
        :param scale:
        """
        assert split in ["train", "dev", "test"]
        indices = get_split(self.split_key)[split]
        self.df = self.get_df().loc[indices]
        if scale:
            self.df = self._scale(self.df)
        super().__init__(split, transform)

class Vo(MultivariateRegressionDataset):

    format = "va"
    variables = constants.VA
    scaling = "tanh"

    @staticmethod
    def _capitalize(w):
        return w[0].upper() + w[1:]


    @classmethod
    def get_df(cls):
        path = utils.get_dataset_dir() / "Vo.csv"
        df = pd.read_csv(path, sep=';', index_col=1)

        # make Nouns upper case
        new_index = []
        for w, pos in df["WORD_CLASS"].items():
            if pos == "N":
                new_index.append(Vo._capitalize(w))
            else:
                new_index.append(w)
        df.index = new_index


        df.index.rename('word', inplace=True)

        dct = {
            'EMO_MEAN': 'valence',
            'AROUSAL_MEAN': 'arousal',
        }

        df = df.rename(columns=dct)
        for c in df.columns:
            if c not in dct.values():
                df.drop(columns=c, inplace=True)

        return df

    def __init__(self, split, transform=None, scale=True):
        assert split in ["train", "dev", "test", "full"]
        self.df = self.get_df()
        if split in ["train", "dev", "test"]:
            indices = get_split("Vo")[split]
            self.df = self.df.loc[indices]
        if scale:
            self.df["valence"] = self.df["valence"].apply(MinMaxScaler(-3, 3, -1, 1))
            self.df["arousal"] = self.df["arousal"].apply(MinMaxScaler(1, 5, -1, 1))
        super().__init__(split, transform)

class Briesemeister(MultivariateRegressionDataset):

    format = "be5"
    variables = constants.BE5
    scaling = "sigmoid"
    path = utils.get_dataset_dir() / "Briesemeister2011.xls"
    split_key = "Briesemeister" #name of the data split file

    @staticmethod
    def _capitalize(w):
        return w[0].upper() + w[1:]

    @classmethod
    def get_df(cls):
        df = pd.read_excel(cls.path, index_col=1)
        df.index.rename('word', inplace=True)

        # introduce proper capitalization
        df.index = [cls._capitalize(w) for w in df.index]

        # rename columns and drop superfluous ones
        dct = {
            'HAP_MEAN': 'joy',
            'ANG_MEAN': 'anger',
            'SAD_MEAN': 'sadness',
            'FEA_MEAN': 'fear',
            'DIS_MEAN': 'disgust'
        }
        df = df.rename(columns=dct)
        for c in df.columns:
            if c not in dct.values():
                df.drop(columns=c, inplace=True)

        return df

    @staticmethod
    def _scale(df):
        scaler = MinMaxScaler(1,5,0,1)
        df = df.applymap(scaler)
        return df

    def __init__(self, split, transform=None, scale=True):
        """
        :param split:
        :param transform:
        :param scale:
        """
        assert split in ["train", "dev", "test"]
        indices = get_split(self.split_key)[split]
        self.df = self.get_df().loc[indices]
        if scale:
            self.df = self._scale(self.df)
        super().__init__(split, transform)


class Kapucu(MultivariateRegressionDataset):

    @classmethod
    def get_df(cls):
        path = utils.get_dataset_dir() / "Kapucu.csv"
        df = pd.read_csv(path, sep=';', index_col=0)
        dc = {
            'ValenceM': 'valence',
            'ArousalM': 'arousal',
            'HappyM': 'joy',
            'AngerM': 'anger',
            'SadM': 'sadness',
            'FearM': 'fear',
            'DisgustM': 'disgust',
        }
        df = df.rename(columns=dc)
        for c in df.columns:
            if c not in dc.values():
                df = df.drop(columns=c)

        # handling of unicode-turkish-i-problem
        # https://stackoverflow.com/questions/48067545/why-does-unicode-implement-the-turkish-i-the-way-it-does
        # https://github.com/emre/unicode_tr
        new_index = []
        for w in df.index:
            w = unicode_tr(w)  # improved unicode encoding
            w = w.lower()
            new_index.append(w)
        df.index = new_index

        df = df[~df.index.duplicated()]
        return df



class Kapucu_VA(Kapucu):

    format = "va"
    variables = constants.VA
    scaling = "tanh"

    @classmethod
    def get_df(cls):
        df = super().get_df()
        df = df[["valence", "arousal"]]
        return df

    def __init__(self, split, transform=None, scale=True):
        assert split in ["train", "dev", "test", "full"]
        self.df = self.get_df()
        if split in ["train", "dev", "test"]:
            indices = get_split("Kapucu")[split]
            self.df = self.df.loc[indices]
        if scale:
            scaler = MinMaxScaler(1, 9, -1, 1)
            self.df[["valence", "arousal"]] = self.df[["valence", "arousal"]].applymap(scaler)
        super().__init__(split, transform)


class Kapucu_BE(Kapucu):

    format = "be5"
    variables = constants.BE5
    scaling = "sigmoid"

    @classmethod
    def get_df(cls):
        df = super().get_df()
        df = df[["joy", "anger", "sadness", "fear", "disgust"]]
        return df

    def __init__(self, split, transform=None, scale=True):
        assert split in ["train", "dev", "test"]
        indices = get_split("Kapucu")[split]
        self.df = self.get_df().loc[indices]
        if scale:
            scaler = MinMaxScaler(0, 100, 0, 1)
            self.df[["joy", "anger", "sadness", "fear", "disgust"]] = self.df[["joy", "anger", "sadness", "fear", "disgust"]].applymap(scaler)
        super().__init__(split, transform)


class ANEW1999(MultivariateRegressionDataset):

    format = "vad"
    variables = constants.VAD
    scaling = "tanh"

    @classmethod
    def get_df(cls):
        path = utils.get_dataset_dir() / "ANEW1999.csv"
        anew = pd.read_csv(path, sep='\t')
        anew.columns = ['word', 'valence', 'arousal', 'dominance']
        anew.set_index('word', inplace=True)
        anew = anew[~anew.index.duplicated()]
        anew = anew[cls.variables]
        return anew

    def __init__(self,
                 split,
                 transform=None,
                 scale=True):
        assert split in ["train", "dev", "test", "full"]
        self.df = self.get_df()
        if split in ["train", "dev", "test"]:
            indices = get_split("ANEW-Stevenson")[split]
            self.df = self.df.loc[indices]
        if scale:
            scaler = MinMaxScaler(1, 9, -1, 1)
            self.df = self.df.applymap(scaler)
        super(ANEW1999, self).__init__(split, transform)


class ANEW1999_VA(ANEW1999):
    format = "va"
    variables = constants.VA



class ANGST2014(MultivariateRegressionDataset):
    format = "vad"
    variables = constants.VAD
    scaling = "tanh"

    @classmethod
    def get_df(cls):
        path = utils.get_dataset_dir() / "ANGST2014.xlsx"
        de_gold = pd.read_excel(path)
        rename = {'VAL_Mean': 'valence',
                  'ARO_Mean_(ANEW)': 'arousal',
                  'DOM_Mean': 'dominance',
                  'G-word': 'word'}

        de_gold.rename(columns=rename, inplace=True)
        for c in de_gold.columns:
            if c not in rename.values():
                de_gold.drop(columns=c, inplace=True)
        de_gold.set_index('word', inplace=True)
        de_gold = de_gold[~de_gold.index.duplicated()]
        return de_gold

    def __init__(self):
        raise NotImplementedError








class Stevenson2007(MultivariateRegressionDataset):

    format = "be5"
    variables = constants.BE5
    scaling = "sigmoid"

    @classmethod
    def get_df(cls):
        path = utils.get_dataset_dir() / "stevenson2007.xls"
        be = pd.read_excel(path, index_col=0)
        rename= {
        'mean_hap': 'joy',
        'mean_ang': 'anger',
        'mean_sad': 'sadness',
        'mean_fear': 'fear',
        'mean_dis': 'disgust'
        }

        be = be[~be.index.isna()]

        be = be.rename(columns=rename)
        for c in be.columns:
            if c not in rename.values():
                be.drop(columns=c, inplace=True)

        be.index = [x.strip() for x in be.index]
        return be

    def __init__(self,
                 split,
                 transform=None,
                 scale=True):

        assert split in ["train", "dev", "test"]
        indices = get_split("ANEW-Stevenson")[split]
        self.df = self.get_df()
        self.df = self.df.loc[indices]
        if scale:
            scaler = MinMaxScaler(1, 5, 0, 1)
            self.df = self.df.applymap(scaler)
        super(Stevenson2007, self).__init__(split, transform)


class NRC(MultiLabelClassificationDataset):

    format = "nrc"
    variables = constants.NRC
    #loss = torch.nn.BCELoss
    scaling = constants.LOGITS

    @classmethod
    def get_df(cls):

        path = utils.get_dataset_dir() / "NRC-Emotion-Lexicon.zip"
        url = "http://sentiment.nrc.ca/lexicons-for-research/NRC-Emotion-Lexicon.zip"
        if not path.is_file():
            print("Downloading NRC Emotion Lexicon ...")
            urlretrieve(url, path)

        with ZipFile(path) as zip:
            with zip.open("NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt") as f:
                df = pd.read_csv(f, sep="\t", names=["word", "emo", "label"], index_col=[0, 1], keep_default_na=False)

        df = pd.DataFrame({var: df.xs(var, level=1)["label"] for var in cls.variables}, dtype=np.int8)
        return df

    def __init__(self, split, transform=None):
        assert split in ["train", "dev", "test"]

        indices = get_split("NRC")[split]
        self.df = self.get_df()
        self.df = self.df.loc[indices]

        super().__init__(split, transform)


class NRC8(NRC):
    """
    Same as NRC but without polarity. (I am afraid, that those labels are far to frequent make the net less "motivated"
    to learn the more difficult plutchik emotions.)
    """
    variables = constants.PLUTCHIK
    format = "nrc8"


class Moors2013(MultivariateRegressionDataset):

    format = "va"
    variables = constants.VA
    scaling = "tanh"

    @classmethod
    def get_df(cls):
        path = utils.get_dataset_dir() / "Moors2013.xlsx"
        df = pd.read_excel(path, header=1)
        df = df[['Words', 'M V', 'M A']]
        df.columns = ["word", "valence", "arousal"]
        df.set_index('word', inplace=True)
        return df

    def __init__(self, split, transform=None, scale=True):

        assert split in ["train", "dev", "test"]
        indices = get_split("Moors2013")[split]
        self.df = self.get_df()
        self.df = self.df.loc[indices]
        if scale:
            scaler = MinMaxScaler(1,7, -1, 1)
            self.df = self.df.applymap(scaler)
        super().__init__(split=split, transform=transform)


