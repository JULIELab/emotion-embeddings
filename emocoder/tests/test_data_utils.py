import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from emocoder.src.data.utils import MultiDataLoaderIterator





def test_MultiDataLoaderIterator():

    # Nested dataset class to simulate interface of used emotion datasets (returning dicts)
    class Dummy_Dataset(torch.utils.data.Dataset):
        def __init__(self, feature_tensor, label_tensor):
            assert len(feature_tensor) == len(label_tensor)
            self.features = feature_tensor
            self.labels = label_tensor

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i):
            return {"features_key": self.features[i],
                    "labels": self.labels[i]}


    ds1 = Dummy_Dataset(torch.Tensor(np.zeros((100, 5))),
                        torch.Tensor(np.zeros((100, 1)))
                        )

    ds2 = Dummy_Dataset(torch.Tensor(np.ones((300, 5))),
                        torch.Tensor(np.ones((300, 3)))
                        )
    dl1 = DataLoader(ds1, batch_size=10, shuffle=True)
    dl2 = DataLoader(ds2, batch_size=10, shuffle=True)
    iter = MultiDataLoaderIterator([dl1, dl2])

    for i, batch in enumerate(iter):
        batch
    assert i + 1 == (len(dl1) + len(dl2))