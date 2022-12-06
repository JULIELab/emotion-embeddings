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
import tarfile
import xml.etree.ElementTree as ET
from emocoder.src.utils import get_split
from emocoder.src import metrics, constants, utils
from typing import Type


from .utils import MinMaxScaler, BaseDataset
from abc import ABC, abstractmethod


class TextDataset(BaseDataset):


    @classmethod
    def score(cls,
              model: torch.nn.Module,
              device: torch.device,
              split: str,
              transform: callable,
              batch_size: int,
              collater: callable = None,
              metric: Type[metrics.Metric] = None):

        if collater is None:
            raise ValueError("Costum collater must be provided for text datasets because of padding.")

        return super().score(model=model,
                             device=device,
                             split=split,
                             transform=transform,
                             batch_size=batch_size,
                             collater=collater,
                             metric=metric)





class MultivarRegressionTextDataset(TextDataset):
    metric = metrics.Pearson
    loss = torch.nn.MSELoss
    problem = constants.MULTIVARIATE_REGRESSION
    performance_key = "mean"
    greater_is_better=True


class MultiClassTextDataset(TextDataset):
    problem = constants.MULTICLASS
    loss = torch.nn.CrossEntropyLoss
    metric = metrics.MulticlassAccuracy
    performance_key = "acc"
    greater_is_better = True


class MultiLabelTextDataset(TextDataset):
    loss = torch.nn.BCEWithLogitsLoss
    metric = metrics.MultiLabelF1
    problem = constants.MULTILABEL
    performance_key = "f1_mean"
    greater_is_better = True


class BinaryTextDataset(TextDataset):
    loss = torch.nn.BCEWithLogitsLoss
    metric = metrics.BinaryAccuracy
    performance_key = "acc"
    greater_is_better = True
    problem = constants.BINARY



class SST_2_Class(BinaryTextDataset):

    format = "pol1"
    variables = constants.POL1
    scaling = constants.LOGITS

    @classmethod
    def get_df(cls):
        path = utils.get_dataset_dir() / "sst.csv"
        df = pd.read_csv(path, sep=",", index_col=0)
        df = df[df.is_binary]
        df = df[["sentence", "two_class", "split"]]
        df.rename(inplace=True, columns={"sentence": "text", "two_class": "polarity"})
        return df

    def __init__(self, split, transform=None):

        assert split in ["train", "dev", "test", "full"]
        self.split = split
        self.transform = transform
        self.df = self.get_df()

        if self.split == "full":
            pass
        elif self.split == "train":
            self.df = self.df[self.df.split == 1]
        elif self.split == "dev":
            self.df = self.df[self.df.split == 3]
        elif self.split == "test":
            self.df = self.df[self.df.split == 2]
        else:
            raise ValueError("Unrecognized split!")


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        sample = {'raw': self.df.text.iloc[idx],
                   'label': self.df[self.variables].iloc[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample



class EmoBank(MultivarRegressionTextDataset):
    """
    @inproceedings{buechel-hahn-2017-emobank,
    title = "{E}mo{B}ank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis",
    author = "Buechel, Sven  and
      Hahn, Udo",
    booktitle = "Proceedings of the 15th Conference of the {E}uropean Chapter of the Association for Computational Linguistics: Volume 2, Short Papers",
    month = apr,
    year = "2017",
    address = "Valencia, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/E17-2092",
    pages = "578--585",
    abstract = "We describe EmoBank, a corpus of 10k English sentences balancing multiple genres, which we annotated with dimensional emotion metadata in the Valence-Arousal-Dominance (VAD) representation format. EmoBank excels with a bi-perspectival and bi-representational design. On the one hand, we distinguish between writer{'}s and reader{'}s emotions, on the other hand, a subset of the corpus complements dimensional VAD annotations with categorical ones based on Basic Emotions. We find evidence for the supremacy of the reader{'}s perspective in terms of IAA and rating intensity, and achieve close-to-human performance when mapping between dimensional and categorical formats.",
    }
    """
    format = "vad"
    variables = constants.VAD
    scaling = constants.TANH


    @classmethod
    def get_df(cls,
               path=utils.get_dataset_dir() / 'EmoBank.zip',
               url="https://github.com/JULIELab/EmoBank/archive/master.zip"):

        # download if necessary
        if not path.is_file():
            print('Downloading data...')
            urlretrieve(url, path)

        with ZipFile(path) as zip:
            with zip.open('EmoBank-master/corpus/emobank.csv') as f:
                df = pd.read_csv(f, index_col=0)

        df = df.rename(columns={"V": "valence", "A": "arousal", "D": "dominance"})

        return df

    def __init__(self,
                 split="full",
                 transform=None,
                 scale=True):
        assert split in ['train', 'dev', 'test', 'full'], 'split must be "train", "dev", "test", or "full".'
        df = self.get_df()
        if not split == "full":
            df = df[df.split == split]

        if scale:
            df[self.variables] = df[self.variables].applymap(MinMaxScaler(1, 5, -1, 1))

        self.df = df
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = {'raw': self.df.text.iloc[idx],
                   'label': self.df[self.variables].iloc[idx]}

        if self.transform:
            # sample['features_key'] = self.transform(sample['raw'])
            sample = self.transform(sample)

        return sample


class SST_5_Class(MultiClassTextDataset):
    """
    @inproceedings{socher-etal-2013-recursive,
    title = "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank",
    author = "Socher, Richard  and
      Perelygin, Alex  and
      Wu, Jean  and
      Chuang, Jason  and
      Manning, Christopher D.  and
      Ng, Andrew  and
      Potts, Christopher",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D13-1170",
    pages = "1631--1642",
    }
    """

    variables = constants.POL5
    scaling = constants.LOGITS



    def __init__(self, split, transform=None):
        """
        Using the https://github.com/JonathanRaiman/pytreebank API.
        :param split:
        :param transform:
        """

        self.transform = transform
        self.split = split
        self.dataset = pytreebank.load_sst()[self.split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        label, sentence = self.dataset[idx].to_labeled_lines()[0]
        sample = {'raw': sentence,
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ISEAR(MultiClassTextDataset):
    format = "izard"
    variables = constants.IZARD
    scaling = constants.LOGITS

    def __init__(self,
                 split,
                 transform=None,
                 url="https://github.com/JULIELab/ISEAR/archive/master.zip",
                 path=utils.get_dataset_dir() /"isear.zip"):

        self.label_encoding = {var:i for i, var in enumerate(self.variables)}

        self.split = split
        self.transform = transform
        assert self.split in ["train", "dev", "test"]

        if not path.is_file():
            print('Downloading data...')
            urlretrieve(url, path)

        # we have an spss-file within in a zip within in a zip.
        # We have to unzip everything into a temp dir to then use pandas to read the data

        with ZipFile(path) as zip:
            with zip.open("ISEAR-master/isear.csv") as f:
                self.df = pd.read_csv(f, index_col="MYKEY")[["Field1", "SIT"]]
                self.df.columns = ["label", "text"]
                indices = get_split("ISEAR")[split]
                self.df = self.df.loc[indices]

        def clean(x):
            x = x.replace("á", "")
            pattern = re.compile(r"\s+")
            x = re.sub(pattern, " ", x)
            x = x.strip()
            return x

        self.df["text"] = self.df.text.apply(clean)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        label, text = self.df.iloc[i]
        label = self.label_encoding[label]
        sample = {"raw": text, "label": label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class SSEC(MultiLabelTextDataset):
    """
    @inproceedings{schuff_annotation_2017,
        address = {Copenhagen, Denmark},
        title = {Annotation, {Modelling} and {Analysis} of {Fine}-{Grained} {Emotions} on a {Stance} and {Sentiment} {Detection} {Corpus}},
        url = {http://www.ims.uni-stuttgart.de/data/ssec},
        booktitle = {Proceedings of the 8th {Workshop} on {Computational} {Approaches} to {Subjectivity}, {Sentiment} and {Social} {Media} {Analysis}},
        publisher = {Association for Computational Linguistics},
        author = {Schuff, Hendrik and Barnes, Jeremy and Mohme, Julian and Padó, Sebastian and Klinger, Roman},
        year = {2017},
        annote = {Presents  tweet emotion data set where the individual ratings are distributed.
    Good survey of existing emotion social media data sets.
     },
        file = {Schuff-2017-Annotation Modelling Analysis.pdf:/Users/sven/Zotero/storage/6RFQLNUM/Schuff-2017-Annotation Modelling Analysis.pdf:application/pdf}
    }
    """
    format = "plutchik"
    variables = constants.PLUTCHIK
    scaling = constants.SIGMOID


    @staticmethod
    def _recode(s: str, var: str) -> int:
        """
        Replacing string label coding with with 1/0
        :param s: The string value to be recoded.
        :param var: The name of the emotion variable in the particular column.
        :return: 1/0.
        """
        if s.lower() == var:
            return 1
        elif s == "---":
            return 0
        else:
            raise ValueError("Expects either the name of the emotion or '---' in each column cell.")

    def __init__(self,
                 split,
                 transform=None,
                 url="http://www.romanklinger.de/ssec/ssec-aggregated-withtext.zip",
                 path=utils.get_dataset_dir() / "SSEC.zip"):

        self.split = split
        self.transform = transform

        if not path.is_file():
            print('Downloading data...')
            urlretrieve(url, path)

        if self.split == "test":
            filepath = ("ssec-aggregated/test-combined-0.0.csv")
        elif self.split == "train":
            filepath = "ssec-aggregated/train-combined-0.0.csv"
        elif self.split == "dev":
            filepath = "ssec-aggregated/test-combined-0.0.csv"
        else:
            raise ValueError()

        with ZipFile(path) as zip:
            with zip.open(filepath) as f:
                self.df = pd.read_csv(f, sep="\t",
                                      names=["anger", "anticipation", "disgust", "fear", "joy",
                                             "sadness", "surprise", "trust"]  + ["text"])
                self.df = self.df[self.variables + ["text"]]

        if self.split == "train":
            self.df.drop([109, 1827], axis=0, inplace=True)  # There are two empty tweets in the dataset
        else:
            indices = get_split("SSEC")[self.split]
            self.df = self.df.loc[indices]


        # Replacing string label coding with with 1/0
        for var in self.variables:
            self.df[var] = self.df[var].apply(lambda x: self._recode(x, var))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        sample = {"raw": self.df.text.iloc[i],
                  "label": self.df[self.variables].iloc[i]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class CVAT(MultivarRegressionTextDataset):
    """
    @inproceedings{yu_building_2016,
        title = {Building {Chinese} {Affective} {Resources} in {Valence}-{Arousal} {Dimensions}.},
        abstract = {An increasing amount of research has recently focused on representing affective states as continuous numerical values on multiple di- mensions, such as the valence-arousal (VA) space. Compared to the categorical approach that represents affective states as several clas- ses (e.g., positive and negative), the dimen- sional approach can provide more fine-grained sentiment analysis. However, affective resources with valence-arousal ratings are still very rare, especially for the Chinese lan- guage. Therefore, this study builds 1) an affective lexicon called Chinese valence-arousal words (CVAW) containing 1,653 words, and 2) an affective corpus called Chinese valence- arousal text (CVAT) containing 2,009 sen- tences extracted from web texts. To improve the annotation quality, a corpus cleanup pro- cedure is used to remove outlier ratings and improper texts. Experiments using CVAW words to predict the VA ratings of the CVAT corpus show results comparable to those ob- tained using English affective resources.},
        booktitle = {Proceedings of {NAACL}-2016},
        author = {Yu, Liang-Chih and Lee, Lung-Hao and Hao, Shuai and Wang, Jin and He, Yunchao and Hu, Jun and Lai, K. Robert and Zhang, Xuejie},
        year = {2016},
    }
    """
    format = "va"
    variables = constants.VA
    problem = constants.MULTIVARIATE_REGRESSION
    metrics = metrics.Pearson
    scaling = constants.TANH
    loss = torch.nn.MSELoss

    def __init__(self,
                 split,
                 transform=None,
                 scale=True):

        # self.vars = list(vars.upper())
        self.transform = transform

        path = utils.get_dataset_dir() / "ChineseEmoBank.zip"
        with ZipFile(path) as zip:
            if split == "train":
                dfs = []
                for i in 1, 2, 3:
                    with zip.open(f"ChineseEmoBank/CVAT/CVAT_{i}.csv") as f:
                        dfs.append(pd.read_csv(f, sep="\t", index_col=0))
                self.df = pd.concat(dfs)
            elif split == "dev":
                with zip.open("ChineseEmoBank/CVAT/CVAT_4.csv") as f:
                    self.df = pd.read_csv(f, sep="\t", index_col=0)
            elif split == "test":
                with zip.open("ChineseEmoBank/CVAT/CVAT_5.csv") as f:
                    self.df = pd.read_csv(f, sep="\t", index_col=0)
            elif split == "full":
                dfs = []
                for i in 1, 2, 3, 4, 5:
                    with zip.open(f"ChineseEmoBank/CVAT/CVAT_{i}.csv") as f:
                        dfs.append(pd.read_csv(f, sep="\t", index_col=0))
                self.df = pd.concat(dfs)
            else:
                raise ValueError()

        self.df.rename({"Valence_Mean": "valence", "Arousal_Mean": "arousal"}, axis=1, inplace=True)

        if scale:
            self.df[self.variables] = self.df[self.variables].applymap(MinMaxScaler(1, 9, -1, 1))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        sample = {"raw": self.df.iloc[i]["Text"],
                  "label": self.df[self.variables].iloc[
                      i]}  # order of self.vars and iloc is important because otherwise the series will be of type object!
        if self.transform:
            sample = self.transform(sample)
        return sample


class AffText(MultivarRegressionTextDataset):
    """
    % ACL Anthology
    @InProceedings{Strapparava07,
    author = "Strapparava, Carlo and Mihalcea, Rada",
    title = "SemEval-2007 Task 14: Affective Text",
    booktitle = "Proceedings of the Fourth International Workshop on Semantic Evaluations (SemEval-2007)",
    year = "2007", publisher = "Association for Computational Linguistics",
    pages = "70--74",
    location = "Prague, Czech Republic",
    url = "http://aclweb.org/anthology/S07-1013" }
    """

    variables = constants.BE6
    scaling = constants.SIGMOID

    @staticmethod
    def _get_df():
        url = "http://web.eecs.umich.edu/~mihalcea/downloads/AffectiveText.Semeval.2007.tar.gz"
        path = utils.get_dataset_dir() /  "AffectiveText.tar.gz"

        if not path.is_file():
            print("Downloading data...")
            urlretrieve(url, path)



        tar = tarfile.open(path, "r:gz")

        dfs = {}
        for split in ["test", "trial"]:

            f = tar.extractfile(f"AffectiveText.{split}/affectivetext_{split}.xml")
            content = f.read().decode("utf8")
            content = content.replace("&", "&amp;")
            tree = ET.fromstring(content)
            texts = {}
            for child in tree.iter("instance"):
                text = child.text
                id = int(child.attrib["id"])
                texts[id] = text
            f = tar.extractfile(f"AffectiveText.{split}/affectivetext_{split}.emotions.gold")
            df = pd.read_csv(f, sep=" ", index_col=0, names=["anger", "disgust", "fear", "joy", "sadness", "surprise"])
            texts = pd.Series(texts)
            df["text"] = texts
            dfs[split] = df

        df = pd.concat([dfs["test"], dfs["trial"]])
        return df

    def __init__(self,
                 split,
                 transform=None,
                 scale=True):





        self.split = split
        self.transform = transform
        #self.vars = self.variables
        self.scale = scale
        if self.split in ["train", "dev", "test"]:
            indices = get_split("AffTextEB")[self.split]
            self.df = self._get_df().loc[indices]
        elif self.split == "full":
            self.df = self._get_df()
        else:
            raise ValueError()


        if scale:
            self.df[self.variables] = self.df[self.variables].applymap(MinMaxScaler(0, 100, 0, 1))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        sample = {"raw": self.df.text.iloc[i],
                  "label": self.df[self.variables].iloc[i]}
        if self.transform:
            sample = self.transform(sample)
        return sample

class AffTextBE5(AffText):
    """
    Normal AffText dataset but without "Surprise".
    """
    format = "be5"
    variables = constants.BE5





#
#
# class SB10k(TextDataset):
#     """
#     @inproceedings{cieliebak-etal-2017-twitter,
#     title = "A Twitter Corpus and Benchmark Resources for {G}erman Sentiment Analysis",
#     author = "Cieliebak, Mark  and
#     Deriu, Jan Milan  and
#     Egger, Dominic  and
#     Uzdilli, Fatih",
#     booktitle = "Proceedings of the Fifth International Workshop on Natural Language Processing for Social Media",
#     month = apr,
#     year = "2017",
#     address = "Valencia, Spain",
#     publisher = "Association for Computational Linguistics",
#     url = "https://www.aclweb.org/anthology/W17-1106",
#     doi = "10.18653/v1/W17-1106",
#     pages = "45--51",
#     abstract = "In this paper we present SB10k, a new corpus for sentiment analysis with approx. 10,000 German tweets. We use this new corpus and two existing corpora to provide state-of-the-art benchmarks for sentiment analysis in German: we implemented a CNN (based on the winning system of SemEval-2016) and a feature-based SVM and compare their performance on all three corpora. For the CNN, we also created German word embeddings trained on 300M tweets. These word embeddings were then optimized for sentiment analysis using distant-supervised learning. The new corpus, the German word embeddings (plain and optimized), and source code to re-run the benchmarks are publicly available.",
#     }
#     """
#
#     URL = "http://4530.hostserv.eu/resources/corpus_v1.0.tsv"
#     PATH = Path.home() / 'data' / 'SB10k'
#
#     @staticmethod
#     def get(path=PATH,
#             url=URL):
#
#         # check if necessary files are there
#         if not path.is_dir():
#             path.mkdir()
#         if not (path / 'meta.tsv').is_file():
#             print('downloading data...')
#             urlretrieve(url, path / 'meta.tsv')
#             print("download complete...")
#
#         ### check if tweets have been downloaded
#         if not (path / "full.tsv"):
#             print('start downloading tweets')
#             # ...
