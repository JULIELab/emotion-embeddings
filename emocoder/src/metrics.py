import numpy as np
import scipy.stats as st
import sklearn
import torch
from typing import Union

class Metric():
    """
    Base class for storing and evaluating model predictions.

    Stores true labels, predicted labels, and optionally dataset ids (relative to dataset instance).

    `Update` updates stored labels, preds and ids. `result` gives the performance based on the stored labels and
    predictions. Attributes `performance_key` and `greater_is_better` are used to determine which of two results is
    better. `to_json` and `from_json` are used to store results for later inspection.
    """
    performance_key: str
    greater_is_better: bool


    def __init__(self,
                 vars: Union[int, list],
                 store_ids=False):
        raise NotImplementedError


    def result(self):
        raise NotImplementedError

    def update(self, true, pred):
        raise NotImplementedError


class Pearson(Metric):

    greater_is_better = True

    def __init__(self,
                 vars: Union[int, list],
                 store_ids=False):

        if isinstance(vars, int):
            self.vars = range(vars)
        elif isinstance(vars, list):
            self.vars = vars
        else:
            raise ValueError("Argument vars must be integer or list of strings")
        if len(self.vars) == 1:
            self.performance_key = self.vars[0]
        elif len(self.vars) > 1:
            self.performance_key = "mean"
        else:
            raise ValueError("Invalid number of variables.")

        self.store_ids = store_ids

        # starting with lazy implementation
        self.true = []
        self.pred = []
        if self.store_ids: self.ids = []

    def update(self, true, pred, ids=None):
        self.true.append(true.cpu().numpy())
        self.pred.append(pred.cpu().numpy())
        if self.store_ids:
            assert ids is not None
            self.ids.append(ids.cpu().numpy())

    def result(self):
        if len(self.vars) > 1:
            true = np.vstack(self.true)
            pred = np.vstack(self.pred)
            rt = [st.pearsonr(true[:, i], pred[:, i])[0] for i in range(len(self.vars))]
            rt.append(np.mean(rt))
            rt = {var: rt[i] for i, var in enumerate(self.vars + ["mean"])}
        else:
            true = np.concatenate(self.true)
            pred = np.concatenate(self.pred)
            rt = {self.vars[0]: st.pearsonr(true,pred)[0]}

        return rt





class MultiLabelF1(Metric):
    """
    Multi-Label
    """
    greater_is_better = True

    def __init__(self,
                 vars: Union[int, list],
                 store_ids=False):

        if isinstance(vars, int):
            self.vars = range(vars)
        elif isinstance(vars, list):
            self.vars = vars
        else:
            raise ValueError("Argument vars must be integer or list of strings")
        if len(self.vars) == 1:
            self.performance_key = f"f1_{self.vars[0]}"
        elif len(self.vars) > 1:
            self.performance_key = "f1_mean"
        else:
            raise ValueError("Invalid number of variables.")

        self.store_ids = store_ids

        # starting with lazy implementation
        self.true = []
        self.pred = []
        if self.store_ids: self.ids = []

    def update(self, true, pred, ids=None):
        """
        Expects logits which are then converted into hard predictions.
        :param true:
        :param pred:
        :return:
        """
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()
        pred  = np.greater(pred, 0) #convert logits into hard predictions
        self.true.append(true)
        self.pred.append(pred)
        if self.store_ids:
            assert ids is not None
            self.ids.append(ids.cpu().numpy())


    def result(self, mean=True, round_decimals = 0):
        true = np.vstack(self.true)
        pred = np.vstack(self.pred)
        prec, rec, f1, support = sklearn.metrics.precision_recall_fscore_support(y_true=true, y_pred=pred,
                                                                                     average=None)  # everything class-wise

        rt = {}
        for i, var in enumerate(self.vars):
            rt[f"prec_{var}"] = prec[i]
            rt[f"rec_{var}"] = rec[i]
            rt[f"f1_{var}"] = f1[i]

        if len(self.vars) > 1:
            rt["prec_mean"] = np.mean(prec)
            rt["rec_mean"] = np.mean(rec)
            rt["f1_mean"] = np.mean(f1)

        return rt



class MulticlassAccuracy(Metric):

    greater_is_better = True
    performance_key = "acc"

    def __init__(self,
                 vars: Union[int, list],
                 store_ids=False):

        if isinstance(vars, int):
            self.vars = range(vars)
        elif isinstance(vars, list):
            self.vars = vars
        else:
            raise ValueError("Argument vars must be integer or list of strings")


        self.store_ids = store_ids

        # starting with lazy implementation
        self.true = []
        self.pred = []
        if self.store_ids: self.ids = []


    def update(self, true, pred, ids=None):
        """
        :param true: Index of correct class.
        :param pred: probablities (or really any kind of unnormalized scores; argmax will be taken anyway) of classes
        :return:
        """
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        self.true.append(true)
        self.pred.append(pred)

        if self.store_ids:
            assert ids is not None
            self.ids.append(ids.cpu().numpy())

    def result(self):
        true = np.concatenate(self.true)
        pred = np.concatenate(self.pred)
        acc = sklearn.metrics.accuracy_score(true, pred)
        return {self.performance_key: acc}


class BinaryAccuracy(Metric):

    greater_is_better = True
    performance_key = "acc"

    def __init__(self,
                 vars: Union[int, list],
                 store_ids=False):


        self.store_ids = store_ids

        # starting with lazy implementation
        self.true = []
        self.pred = []
        if self.store_ids: self.ids = []


    def update(self, true, pred, ids=None):
        """
        :param true: Index of correct class.
        :param pred: probablities (or really any kind of unnormalized scores; argmax will be taken anyway) of classes
        :return:
        """
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()
        pred = np.greater(pred, 0) # transform logits into hard predictions
        self.true.append(true)
        self.pred.append(pred)

        if self.store_ids:
            assert ids is not None
            self.ids.append(ids.cpu().numpy())

    def result(self):
        true = np.concatenate(self.true)
        pred = np.concatenate(self.pred)
        acc = sklearn.metrics.accuracy_score(true, pred)
        return {self.performance_key: acc}






