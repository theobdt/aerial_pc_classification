import torch
from torch.utils.data import Dataset
import numpy as np

from utils.ply import ply2dict
from sklearn.neighbors import KDTree


class AerialPointDataset(Dataset):
    def __init__(self, input_file, features, n_neighbors, all_labels=False):
        "Initialization"
        data = ply2dict(input_file)
        try:
            all_features = ["x", "y", "z"] + features
            X = np.vstack([data[f] for f in all_features]).T
        except KeyError:
            print(f"ERROR: Input features {features} not recognized")
            return
        labels = data["labels"]

        self.index = np.arange(X.shape[0])
        if not all_labels:
            X, labels = self.filter_labels(X, labels)

        self.X = torch.from_numpy(X)
        self.labels = torch.from_numpy(labels)
        self.n_samples = self.labels.shape[0]
        tree = KDTree(self.X[:, :3])
        _, self.neighbors_idx = tree.query(
            self.X[:, :3], k=n_neighbors, sort_results=True
        )

    def filter_labels(self, X, labels):
        new_labels = convert_labels(labels)
        mask = new_labels >= 0
        self.index = self.index[mask]
        return X[mask], new_labels[mask]

    def __getitem__(self, index):
        point = self.X[index].view(1, -1)
        neighbors = self.X[self.neighbors_idx[index]]
        sequence = torch.cat((point, neighbors), 0)

        return sequence, self.labels[index]

    def __len__(self):
        return self.n_samples


def convert_labels(labels):
    """Convert 9-labels to 4-labels as follows:
    0 Powerline              -> -1 Other
    1 Low vegetation         -> 0 GLO
    2 Impervious surfaces    -> 0 GLO
    3 Car                    -> -1 Other
    4 Fence/Hedge            -> -1 Other
    5 Roof                   -> 1 Roof
    6 Facade                 -> 2 Facade
    7 Shrub                  -> 3 Vegetation
    8 Tree                   -> 3 Vegetation
    """
    LABELS_MAP = {0: -1, 1: 0, 2: 0, 3: -1, 4: -1, 5: 1, 6: 2, 7: 3, 8: 3}
    return np.vectorize(LABELS_MAP.get)(labels)
