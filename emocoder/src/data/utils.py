#from git import Repo
import random
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
from emocoder.src.utils import get_split
from emocoder.src.data import vectors
from emocoder.src import constants, metrics

from typing import Type
from abc import ABC, abstractmethod

SPLITS = ["train", "dev", "test", "full"]




class Tokenize_Transformer_with_Padding():

    def __init__(self, tokenizer, pad=True, max_len=100, return_tensors='pt'):
        self.tokenizer = tokenizer
        self.pad = pad,
        self.return_tensors=return_tensors
        self.max_len = max_len

    def __call__(self, sample):
        sample['features_key'] = self.tokenizer.encode(sample['raw'],
                                                   pad_to_max_length=self.pad,
                                                   return_tensors=self.return_tensors,
                                                   max_length=self.max_len).squeeze() #watch out for the squeeze, otherwise additional dimension screws up batching!
        sample['label'] = sample['label'].to_numpy()
        return sample



class Tokenize_Transformer():
    """
    Data Tranform for BERT-style tokenization. Added features_key are vocab indices.
    """

    def __init__(self, tokenizer, label_to_numpy=True, label_to_one_hot=False, n_classes=None):
        """
        :param tokenizer: Callable. Produces list of vocabulary indices.
        :param label_to_numpy: Boolean. should be False when labels are integer.
        :param label_to_one_hot: Boolean. For classification problems. Converts integer encoding of correct class
               into one-hot encoding. Expects n_classes to be set.
        :param n_classes. Int. Number of classes for one-hot encoding.
        """
        self.tokenizer = tokenizer
        self.to_numpy = label_to_numpy
        self.label_to_one_hot = label_to_one_hot
        self.n_classes = n_classes

    def __call__(self, sample):
        sample['features'] = self.tokenizer.encode(sample['raw'])
        if self.label_to_one_hot: # assumes label to be integer
            tmp = np.zeros((self.n_classes))
            tmp[sample["label"]] = 1
            sample["label"] = tmp

        if self.to_numpy:
            sample['label'] = sample['label'].to_numpy()
        return sample


class Embedding_Lookup_Transform():
    """
    Data Transform expects datasets of single words and adds embedding features_key.
    """

    def __init__(self, embeddings, limit=None, dim=None):
        """

        :param embeddings: dict-like. String -> 1D np.array
        """
        if isinstance(embeddings, dict):
            self.embs = embeddings
            self.dim = dim
            if self.dim is None:
                raise ValueError("Dimension of embeddings must be specified when passed as dict")
        elif isinstance(embeddings, str):
            self.embs = vectors.EMBEDDINGS[embeddings](limit=limit)
            self.dim = self.embs.dims

        elif issubclass(embeddings, vectors.Embedding_Model):
            self.embs = embeddings()
            self.dim = self.embs.dims

        else:
            raise ValueError("Embeddings must either be passed as str, dict, or embedding subclass")

    def __call__(self, sample):
        """

        :param sample: dict-like
        :return:
        """
        word = sample['raw']
        if word in self.embs:
            sample['features'] = torch.tensor(self.embs[word], dtype=torch.float32)
        else:
            sample['features'] = torch.tensor(np.zeros(self.dim), dtype=torch.float32)

        sample['labels'] = torch.tensor(sample['labels'], dtype=torch.float32)

        return sample

def get_text_transform_and_collater(dataset_class, tokenizer):
    """
    Returns a collater and data transform object based on the dataset class and the used tokenizer.
    This is an awkward piece of logic that is necessary in every text experiment, because only text experiments
    require a custom collater.

    THE PROBLEM is that for regression, multi-label and binary classification problems,
    the data type of the labels always defaults to (vectors of) float32. But multiclass classification is a single
    Long for some reason *shrug* For example,  in a multiclass problems a label can be '2' but in multilabel problems
    it would be '[0., 0., 1., 0.]'. If I remember correctly, this  behavior is even built into the respective
    pytorch loss functions, so there is really no feasible way around it. This function at least saves some boilerplate
    code that would otherwise appear in each of the texts experiments...

    :param dataset_class: The class of the datasets to run experiments on.
    :param tokenizer: The (transformer-) tokenizer.
    :return: Tuple of a transform and a collater instance.
    """

    if dataset_class.problem in [constants.MULTIVARIATE_REGRESSION, constants.MULTILABEL, constants.BINARY]:
            transform = Tokenize_Transformer(tokenizer=tokenizer)
            collater = Collater(padding_symbol=tokenizer.pad_token_id,
                        num_labels=len(dataset_class.variables),
                        label_dtype=torch.float32)
    elif dataset_class.problem == constants.MULTICLASS:
        transform = Tokenize_Transformer(tokenizer=tokenizer, label_to_numpy=False)
        collater = Collater(padding_symbol=tokenizer.pad_token_id,
                                 num_labels=len(dataset_class.variables),
                                 label_dtype=torch.long)
    else:
        raise ValueError

    return transform, collater







class Collater():

    def __init__(self, padding_symbol, num_labels, label_dtype,
                 additional_keys:list=None # samples keys to return batch wise as list
                 ):
        """
        Receives individual samples provided by dataloader and returns batches of equally-sized, padded feature and
        label tensors. Particularly important for text data.
        :param padding_symbol: Padding symbol recognized by the model. In case of transformer, this is ususally stored
        in the tokenizer class.
        :param num_labels: The dimensionality of labels. Corresponds to the number of classes in multi-class problems
         and the number of variables in multivariate regression. Otherwise its 1.
        :param label_dtype:
        :param additional_keys:
        """
        self.padding_symbol = padding_symbol
        self.num_labels = num_labels
        self.label_dtype = label_dtype
        self.additional_keys = additional_keys

    def __call__(self, samples, *args, **kwargs):
        batch = {}
        batch['features'] = [torch.tensor(x['features'], dtype=torch.long) for x in samples]
        lens = [len(x) for x in batch['features']]
        max_len = max(lens)
        tmp = []
        for i, features in enumerate(batch['features']):
            curr_len = len(features)
            padded = torch.nn.functional.pad(features, (0, max_len - curr_len), 'constant', self.padding_symbol)
            tmp.append(padded)
        batch['features'] = torch.stack(tmp)
        if self.num_labels > 1: #array valued labels
            batch['labels'] = torch.stack([torch.tensor(s['label'], dtype=self.label_dtype) for s in samples])
        elif self.num_labels == 1 :
            batch['labels'] = torch.tensor([s['label'] for s in samples], dtype=self.label_dtype)
        elif self.num_labels == 0: # no labels given
            pass
        else:
            raise ValueError()
        batch['raw'] = [s['raw'] for s in samples]

        if self.additional_keys:
            for k in self.additional_keys:
                batch[k] = [s[k] for s in samples]

        return batch



class MinMaxScaler():

    def __init__(self, oldmin, oldmax, newmin, newmax):
        self.oldmin = oldmin
        self.oldmax = oldmax
        self.newmin = newmin
        self.newmax = newmax

    def __call__(self, x):
        return ((self.newmax-self.newmin)*(x-self.oldmin))/(self.oldmax-self.oldmin)+self.newmin



class MultiDataLoaderIterator():

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.dataloaderiters = [iter(dl) for dl in self.dataloaders]
        batch_list = []
        for i,dl in enumerate(self.dataloaders):
            batch_list = batch_list + ([i]*len(dl))
        random.shuffle(batch_list)
        self.batch_list = batch_list

        self.index = 0

    def __len__(self):
        return len(self.batch_list)

    def __next__(self):
        if self.index >= len(self.batch_list):
            raise StopIteration
        else:
            dataloader_index = self.batch_list[self.index]
            batch = self.dataloaderiters[dataloader_index].next()
            batch["dataloader"] = self.dataloaders[dataloader_index]
            self.index += 1
            return batch

    def __iter__(self):
        return self


class MultiDataLoaderIterator2():
    """
    Samples dataloader first. Thus items from smaller datasets are chosen more frequently.
    """
    def __init__(self, dataloaders, batches_per_epoch):
        self.dataloaders = dataloaders
        self.dataloaderiters = [iter(dl) for dl in self.dataloaders]
        self.epoch_size = batches_per_epoch
        self.batches = 0

    def __len__(self):
        return self.epoch_size

    def __next__(self):
        if self.batches >= self.epoch_size:
            raise StopIteration
        else:
            dl_index = random.randint(0, len(self.dataloaders)-1)
            try:
                batch = self.dataloaderiters[dl_index].next()
            except StopIteration:
                self.dataloaderiters[dl_index] = iter(self.dataloaders[dl_index])
                batch = self.dataloaderiters[dl_index].next()

            batch["dataloader"] = self.dataloaders[dl_index]
            self.batches += 1
            return batch

    def __iter__(self):
        self.batches = 0
        return self

class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets classes, TextDataset, WordDataset, and ImageDataset, but not
    mapping dataset.
    """

    format: str
    variables: list #set of emotional variables from constants
    #loss: Type[torch.nn.Module] # torch loss function class
    metric: Type[metrics.Metric] # emocoder.src.metrics.Metric
    scaling: str # from constants. How should the model outputs be scaled to be understood ( / in the right value range)
    # for the metric and the dataset (i.e., Multivariate Regression datasets for VAD are tanh scaled,
    # mulitvariate regression datasets for BE5 are sigmoid-scaled, as are Multilabel datasets, Multiclass datasets are
    # not scaled at all (they expect logits as model outputs) because that's what works best for metrics and available
    # pytorch loss (i.e., there is no simple cross entropy loss on top of softmax).
    problem: str # from constants
    loss: Type[torch.nn.Module]
    performance_key: str
    greater_is_better: bool

    #### This is currently buggy in pycharm: https://intellij-support.jetbrains.com/hc/en-us/community/posts/360002090139-PyCharm-Call-to-init-of-super-class-is-missed

    @abstractmethod
    def __init__(self, split=None, transform=None):
        assert self.problem is not None
        assert self.variables is not None
        assert self.loss is not None
        assert self.metric is not None
        assert self.format in constants.FORMATS

        self.split = split
        self.transform = transform

        assert self.split in ['train', 'dev', 'test', 'full'], 'split must be "train", "dev", "test", or "full".'

    @classmethod
    def score(cls,
              model: torch.nn.Module,
              device: torch.device,
              split: str,
              transform: callable,
              batch_size: int=32,
              collater: callable = None,
              metric: Type[metrics.Metric] = None):

        metric_cls = cls.metric if metric is None else metric
        metric = metric_cls(cls.variables)
        loss_fn = cls.loss()
        running_loss = torch.tensor(0.)


        ds = cls(split=split,
                 transform=transform)
        dl = DataLoader(dataset=ds,
                        batch_size=batch_size,
                        collate_fn=collater,
                        shuffle=False)
        model.eval()
        with torch.no_grad():

            for batch in dl:
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)
                preds = model(features)
                running_loss += loss_fn(preds, labels)
                metric.update(labels, preds)
            running_loss = running_loss.item() / len(ds)
        return metric, running_loss