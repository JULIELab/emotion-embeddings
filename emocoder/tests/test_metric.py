import numpy as np
import math
from emocoder.src.metrics import Pearson
import scipy.stats as st
import torch


def test_pearson():

    # creating random number series
    true = np.random.uniform(low=1.0, high=10.0, size=(1000,3))
    residuals = np.random.uniform(low=0.0, high = 5.0, size=(1000,3))
    pred = true + residuals
    varnames = ["var1", "var2", "var3"]

    # create metric object and feed data batchwise

    metric = Pearson(varnames, False)
    for i in range(10):
        i_start = i * 100
        i_stop = i_start + 100
        metric.update(true=torch.tensor(true[i_start:i_stop]), pred=torch.tensor(pred[i_start:i_stop]))
    actual = [metric.result()[i] for i in varnames]
    expected = [st.pearsonr(true[:,i], pred[:,i])[0] for i in range(len(varnames))]
    assert np.allclose(actual, expected)


    # Repeat with only one variable
    true = true[:,0]
    pred = pred[:,0]
    varnames = varnames[:1]

    metric = Pearson(varnames, False)
    for i in range(10):
        i_start = i * 100
        i_stop = i_start + 100
        metric.update(true=torch.tensor(true[i_start:i_stop]), pred=torch.tensor(pred[i_start:i_stop]))
    actual = metric.result()["var1"]
    expected = st.pearsonr(true, pred)[0]
    assert math.isclose(actual, expected)


# TODO Write test for remainder of the metrics





