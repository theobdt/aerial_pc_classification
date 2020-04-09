import torch
from torch.utils.data import Dataset
import numpy as np

from utils.ply import ply2dict
from sklearn.neighbors import KDTree


LABELS_MAP = {0: 4, 1: 0, 2: 0, 3: 4, 4: 4, 5: 1, 6: 2, 7: 3, 8: 3}


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

        if not all_labels:
            X, labels = filter_labels(X, labels)

        self.X = torch.from_numpy(X)
        self.labels = torch.from_numpy(labels)
        self.n_samples = self.labels.shape[0]
        tree = KDTree(self.X[:, :3])
        _, self.inds = tree.query(
            self.X[:, :3], k=n_neighbors, sort_results=True
        )

    def __getitem__(self, index):
        point = self.X[index].view(1, -1)
        neighbors = self.X[self.inds[index]]
        sequence = torch.cat((point, neighbors), 0)

        return sequence, self.labels[index]

    def __len__(self):
        return self.n_samples


def filter_labels(X, labels):
    new_labels = np.vectorize(LABELS_MAP.get)(labels)
    mask = new_labels != 4
    return X[mask], new_labels[mask]


# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     import time

#     ds = AerialPointDataset("data/features/vaihingen3D_train.ply", 20)
#     loader = DataLoader(ds, batch_size=32, num_workers=4, shuffle=True)

#     print("start loading")
#     now = time.time()
#     for i, (x, y) in enumerate(loader):
#         pass
#     elapsed = time.time() - now
#     print(elapsed)
#     print(f"avg: {elapsed/ len(loader)}")

#     print(len(loader))
