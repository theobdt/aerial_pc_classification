import torch
from torch.utils.data import DataLoader
import argparse
import os
import yaml
from tqdm import tqdm
import numpy as np

from utils.dataloader import AerialPointDataset
from utils.ply import ply2dict, dict2ply
from models import BiLSTM


parser = argparse.ArgumentParser(description="Training")

parser.add_argument(
    "--files", "-f", type=str, nargs="+", help="Path to point cloud file"
)
parser.add_argument(
    "--ckpt", type=str, help="Path to the checkpoint folder",
)
parser.add_argument(
    "--batch_size", type=int, default=1000, help="Batch size",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="Number of workers for dataloading",
)
parser.add_argument(
    "--prediction_folder",
    type=str,
    default="data/predictions",
    help="Path to the prediction folder",
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use : {device}")


# Load checkpoint
path_ckpt = os.path.normpath(args.ckpt)
print(f"Loading checkpoint: {path_ckpt}")
path_config = os.path.join(path_ckpt, "config.yaml")
path_ckpt_dict = os.path.join(path_ckpt, "ckpt.pt")
checkpoint = torch.load(path_ckpt_dict, map_location=device)

# Create prediction folder
ckpt_id = os.path.basename(path_ckpt)
ckpt_prediction_folder = os.path.join(args.prediction_folder, ckpt_id)
os.makedirs(ckpt_prediction_folder, exist_ok=True)

# Load model config
with open(path_config, "r") as f:
    config = yaml.safe_load(f)

# Load model
n_features = len(config["data"]["features"])
n_classes = 4
if config["data"]["all_labels"]:
    n_classes = 9
print(f"Num classes: {n_classes}\n")

print("Loading model..", end=" ", flush=True)
model = BiLSTM(n_features, n_classes, **config["network"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("DONE")


def predict(loader, len_dataset):
    predictions = torch.empty(len_dataset, dtype=torch.int32, device=device)
    with torch.no_grad():
        start = 0
        for (sequence, label) in tqdm(loader, desc="* Processing point cloud"):
            sequence = sequence.to(device)
            label = label.to(device)
            # label = label.type(dtype=torch.long)

            # compute predicted classes
            output = model(sequence)
            classes = torch.max(output, 1).indices

            # fill predictions
            seq_len = sequence.shape[0]
            predictions[start : start + seq_len] = classes
            start += seq_len

    return predictions.numpy()


for path_ply in args.files:
    print(f"\nProcessing file: {path_ply}")
    print("* Preparing dataloader..", end=" ", flush=True)
    dataset = AerialPointDataset(path_ply, **config["data"])
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print("DONE")

    # Create and fill point cloud field
    data = ply2dict(path_ply)
    true_labels = data["labels"]
    n = len(true_labels)
    predictions = np.ones(n, dtype=np.int32)
    raw_predictions = predict(loader, len(dataset))
    predictions[dataset.index] = raw_predictions
    errors = np.logical_and(predictions >= 0, predictions != true_labels)
    data["predictions"] = predictions
    data["errors"] = errors.astype(np.uint8)

    # Save point cloud
    filename = os.path.basename(path_ply)
    path_prediction = os.path.join(ckpt_prediction_folder, filename)
    if dict2ply(data, path_prediction):
        print(f"* Predictions PLY file saved to: {path_prediction}")
