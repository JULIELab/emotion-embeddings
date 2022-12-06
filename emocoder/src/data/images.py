from pathlib import Path
from typing import Type
import pandas as pd
from torch.utils.data import Dataset
import torch
from urllib.request import urlretrieve
from zipfile import ZipFile
import io
import numpy as np
import os
import torchvision
from torchvision import transforms
from PIL import Image
import json
import pytreebank
import re
import tempfile
from emocoder.src.utils import get_split
from emocoder.src import utils, constants, metrics
from PIL import Image

from .utils import MinMaxScaler, BaseDataset
from abc import ABC, abstractmethod


def get_ResNet_Preprocessor(data_augmentation:bool):
    if not data_augmentation: # i.e., for testing
        return  transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    # transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    else:
        return transforms.Compose([
                                    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)), #default: scale=(.08, 1.0)
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])




class ImageDataset(BaseDataset):
    pass


class YOU2017(ImageDataset):
    """
    Load a image dataset with predefined lables. It is important to know, that the data need to be organised in a certain
    structure, thus this class can only be used for discrete annotations, such as base emotions:
    root/smile/xxx.png
    root/smile/xxy.jpeg
    root/smile/xxz.png

    root/sad/123.jpg
    root/sad/nsdf3.png
    root/sad/asd932_.png
    """



    path = utils.get_dataset_dir() / "Flickr"

    format = "be_flickr"
    problem = constants.MULTICLASS
    loss = torch.nn.CrossEntropyLoss
    variables = constants.BE_FLICKR
    metric = metrics.MulticlassAccuracy
    performance_key = "acc"
    greater_is_better = True

    def __init__(self,
                 split,
                 transform=None):

        super().__init__(split, transform)

        dataset = torchvision.datasets.ImageFolder(root=self.path, transform=self.transform)

        if self.split != 'full':
            indices = get_split("YOU2017")[self.split]
        else:
            indices = list(range(0, len(dataset)))

        self.data = torch.utils.data.Subset(dataset, indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (image, lable) = self.data[index]

        sample = {'features': image,
                  'labels': lable}

        return sample




class FER2013Base(ImageDataset):
    """
    https://arxiv.org/abs/1307.0414

    Load a image dataset with predefined lables. It is important to know, that the data need to be organised in a certain
    structure, thus this class can only be used for discrete annotations, such as base emotions:
    root/smile/xxx.png
    root/smile/xxy.jpeg
    root/smile/xxz.png

    root/sad/123.jpg
    root/sad/nsdf3.png
    root/sad/asd932_.png
    """

    path = utils.get_dataset_dir() / "fer2013" / "fer2013+vad.csv"

    #problem = constants.MULTICLASS
    #loss = torch.nn.CrossEntropyLoss
    #variables = constants.BE_FER13
    #metric = metrics.MulticlassAccuracy

    @staticmethod
    def string_to_image(s):
        tmp = s.strip().split()
        tmp = [int(x) for x in tmp]
        tmp = np.array(tmp, dtype=np.uint8).reshape((48, 48))
        return tmp


    def _get_images(self, df):
        images = [self.string_to_image(s) for s in df.pixels]
        images = [Image.fromarray(a, mode="L").convert("RGB") for a in images]
        return images

    def __init__(self,
                 split,
                 transform=None):

        super().__init__(split, transform)

        df = pd.read_csv(self.path, index_col=0)
        if self.split == "train":
            df = df[df.Usage == "Training"]
        elif self.split == "dev":
            df = df[df.Usage == "PublicTest"]
        elif self.split == "test":
            df = df[df.Usage == "PrivateTest" ]
        elif self.split == "full":
            pass
        else:
            raise ValueError()

        self.labels = self.get_ratings(df) #needs to implemented in Subclasses!
        self.images = self._get_images(df)
        self.indices = np.array(df.index)

        assert len(self.labels) == len(self.images)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        sample = {'features': self.images[i],
                  'labels': self.labels[i],
                  'id': self.indices[i]}

        if self.transform:
            sample["features"] = self.transform(sample["features"])

        return sample


class FER2013(FER2013Base):
    format = "be_fer13"
    problem = constants.MULTICLASS
    loss = torch.nn.CrossEntropyLoss
    variables = constants.BE_FER13
    metric = metrics.MulticlassAccuracy
    performance_key = "acc"
    greater_is_better = True

    @classmethod
    def get_ratings(cls, df):
        """
        Extracts the respective ratings (either basic emotion classes or VAD)
        :param df:
        :return:
        """
        return np.asarray(df.emotion)

class FER2013Vad(FER2013Base):
    format = "vad"
    problem = constants.MULTIVARIATE_REGRESSION
    loss = torch.nn.MSELoss
    variables = constants.VAD
    metric = metrics.Pearson
    performance_key = "mean"
    greater_is_better = True

    @staticmethod
    def scale(df):
        scaler = MinMaxScaler(0, 1, -1, 1)
        df = df.applymap(scaler)
        return df

    @classmethod
    def get_ratings(cls, df):
        df =df [cls.variables]
        df = cls.scale(df)
        return np.asarray(df, dtype=np.float32)


class IAPS2008(ImageDataset):
    path = utils.get_dataset_dir() / "IAPS"

    format = "vad"
    problem = constants.MULTIVARIATE_REGRESSION
    loss = torch.nn.MSELoss
    variables = constants.VAD
    metric = metrics.Pearson
    performance_key = "mean"
    greater_is_better = True


    def __init__(self,
                 split,
                 transform=None,
                 path=path,
                 scale=True):

        super().__init__(split, transform)


        image_folder = 'IAPS 1-20 Images'
        lable_file = 'IAPS Tech Report/AllSubjects_1-20.txt'


        data = {}
        dataset = pd.read_csv(path / lable_file, skiprows=6, sep='\t', na_values='.').astype({'IAPS': str})

        if self.split != 'full':
            indices = get_split("IAPS2008")[self.split]
            dataset = dataset.loc[indices]

        if scale:
            scaler = MinMaxScaler(1, 9, -1, 1)

        for index, row in dataset.iterrows():
            valence = row['valmn']
            dominance = row['dom1mn'] if not np.isnan(row['dom1mn']) else row['dom2mn']
            arousal = row['aromn']

            if scale:
                valence = scaler(valence)
                dominance = scaler(dominance)
                arousal = scaler(arousal)

            # the names and image suffixes are not consistent for a handfull of examples
            imagepath = os.path.join(path, image_folder, row['IAPS'].replace('.0', '.jpg'))
            if not os.path.exists(imagepath):
                imagepath = imagepath.replace('jpg', 'JPG')
                if not os.path.exists(imagepath):
                    imagepath = os.path.join(path, image_folder, row['IAPS'] + '.jpg')
                    if not os.path.exists(imagepath):
                        imagepath = imagepath.replace('jpg', 'JPG')

            # try:
            image = Image.open(imagepath)
            if self.transform:
                image = self.transform(image)
            else:
                image = None
            label = np.array([np.float32(valence), np.float32(arousal), np.float32(dominance)])
            data[index] = {'label': label,
                           'image': image}
            # except IOError:
            #     print('Was not able to read: {}\n Check path and corresponding dataset.', imagepath)

        self.data = data
        self._df = dataset


    def __getitem__(self, index):
        keylist = sorted(self.data.keys())
        return {"features": self.data.get(keylist[index]).get('image'),
                "labels": self.data.get(keylist[index]).get('label')}

    def __len__(self):
        return len(self.data)



class AffectNet2019Base(ImageDataset):
    """

    """

    path = utils.get_dataset_dir() / "AffectNet"

    def string_to_image(self, s):
        imagepath = self.path / "ManuallyAnnotated" / s
        image = Image.open(imagepath)
        # if self.transform:
        #     image = self.transform(image)
        # else:
        #     image = None
        return image


    # def _get_images(self, df):
    #     images = [self.string_to_image(s) for s in df.subDirectory_filePath]
    #     return images

    def __init__(self,
                 split,
                 transform=None):

        super().__init__(split, transform)

        # dataset_train = pd.read_csv(self.path / 'Labels/ManuallyAnnotated/training.csv', sep=',')
        # dataset_test = pd.read_csv(self.path / 'Labels/ManuallyAnnotated/validation.csv', sep=',')
        # dataset_test.columns = dataset_train.columns
        #
        # df = pd.concat([dataset_train, dataset_test]).drop(columns=['facial_landmarks'])

        if self.split == "train":
            self.df = pd.read_csv(self.path / 'Labels/ManuallyAnnotated/training.csv', sep=',').drop(columns=['facial_landmarks'])
        elif self.split == "dev":
            self.df = pd.read_csv(self.path / 'Labels/ManuallyAnnotated/validation.csv', sep=',')
            self.df.columns = ["subDirectory_filePath", "face_x", "face_y", "face_width", "face_height",
                               "facial_landmarks", "expression", "valence", "arousal"]
            self.df = self.df.drop(columns=['facial_landmarks'])
            half_index = int(len(self.df) /2)
            self.df = self.df.iloc[:half_index]
        elif self.split == "test":
            self.df = pd.read_csv(self.path / 'Labels/ManuallyAnnotated/validation.csv', sep=',')
            self.df.columns = ["subDirectory_filePath","face_x","face_y","face_width","face_height","facial_landmarks",
                               "expression","valence","arousal"]
            self.df = self.df.drop(columns=['facial_landmarks'])
            half_index = int(len(self.df) /2)
            self.df = self.df.iloc[half_index:]
        elif self.split == "full":
            dataset_train = pd.read_csv(self.path / 'Labels/ManuallyAnnotated/training.csv', sep=',')
            dataset_test = pd.read_csv(self.path / 'Labels/ManuallyAnnotated/validation.csv', sep=',')
            dataset_test.columns = dataset_train.columns
            self.df = pd.concat([dataset_train, dataset_test]).drop(columns=['facial_landmarks'])
        else:
            raise ValueError(f"Unrecognized data split: {self.split}")

        # remove the following classes form the dataset: 8: None, 9: Uncertain, 10: No - Face
        self.df = self.df[self.df.expression < 8]


        self.labels = self.get_ratings(self.df) #needs to implemented in Subclasses!
        # self.images = self._get_images(df)
        self.images = self.df.subDirectory_filePath
        self.indices = list(self.df.index)


        assert len(self.labels) == len(self.images)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        sample = {'features': self.string_to_image(self.images.iloc[i]),
                  'labels': self.labels[i],
                  'id': self.indices[i]}

        if self.transform:
            sample["features"] = self.transform(sample["features"])

        return sample


class AffectNet2019_BE(AffectNet2019Base):
    format = "be_affectnet"
    problem = constants.MULTICLASS
    loss = torch.nn.CrossEntropyLoss
    variables = constants.BE_AFFECTNET
    metric = metrics.MulticlassAccuracy
    greater_is_better = True
    performance_key = "acc"
    greater_is_better = True


    @classmethod
    def get_ratings(cls, df):
        """
        Extracts the respective ratings (either basic emotion classes or VAD)
        :param df:
        :return:
        """
        return np.asarray(df.expression)

class AffectNet2019_VA(AffectNet2019Base):
    format = "va"
    problem = constants.MULTIVARIATE_REGRESSION
    loss = torch.nn.MSELoss
    variables = constants.VA
    metric = metrics.Pearson
    performance_key = "mean"
    greater_is_better = True

    @staticmethod
    def scale(df):
        scaler = MinMaxScaler(-1, 1, -1, 1)
        df = df.applymap(scaler)
        return df

    @classmethod
    def get_ratings(cls, df):
        df = df[cls.variables]
        df = cls.scale(df)
        return np.asarray(df, dtype=np.float32)



