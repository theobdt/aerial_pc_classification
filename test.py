import torch
from torch.utils.data import DataLoader
import argparse
import os
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
import sklearn.metrics as skm

from utils.dataloader import AerialPointDataset, convert_labels
from utils.ply import ply2dict, dict2ply
from models import BiLSTM

NAMES_9 = [
    "Powerline",
    "Low veg.",
    "Imp. surf.",
    "Car",
    "Fence",
    "Roof",
    "Facade",
    "Shrub",
    "Tree",
]

NAMES_4 = ["GLO", "Roof", "Facade", "Vegetation"]


parser = argparse.ArgumentParser(description="Training")

parser.add_argument(
    "--files", "-f", type=str, nargs="+", help="Path to point cloud file"
)
parser.add_argument(
    "--ckpt", type=str, help="Path to the checkpoint folder",
)
parser.add_argument(
    "--prefix_path", type=str, default="", help="Path prefix",
)
parser.add_argument(
    "--batch_size", type=int, default=1000, help="Batch size",
)
parser.add_argument(
    "--weighted_avg",
    action="store_true",
    help="Average metrics with support as weight",
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
path_ckpt = os.path.join(args.prefix_path, os.path.normpath(args.ckpt))
print(f"Loading checkpoint: {path_ckpt}")
path_config = os.path.join(path_ckpt, "config.yaml")
path_ckpt_dict = os.path.join(path_ckpt, "ckpt.pt")
checkpoint = torch.load(path_ckpt_dict, map_location=device)

# Create prediction folder
ckpt_id = os.path.basename(path_ckpt)
ckpt_prediction_folder = os.path.join(
    args.prefix_path, args.prediction_folder, ckpt_id
)
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

            # compute predicted classes
            output = model(sequence)
            classes = torch.max(output, 1).indices

            # fill predictions
            seq_len = sequence.shape[0]
            predictions[start : start + seq_len] = classes
            start += seq_len

    return predictions.cpu().numpy()


def evaluate(y_true, y_pred, names, weighted_avg=False):
    labels = np.arange(len(names))

    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
    totals = np.sum(cm, axis=1)
    cm = np.hstack((cm, totals.reshape(-1, 1)))
    cm = np.vstack((cm, np.sum(cm, axis=0, keepdims=True)))

    metrics = skm.precision_recall_fscore_support(
        y_true, y_pred, labels=labels
    )
    metrics = np.vstack(metrics[:-1]).T
    if weighted_avg:
        avg_metrics = totals @ metrics / np.sum(totals)
        name_last_row = "Total/Weighted Avg"
    else:
        avg_metrics = np.mean(metrics, axis=0)
        name_last_row = "Total/Avg"
    metrics = np.vstack((metrics, avg_metrics))

    all_data = np.hstack((cm, metrics))

    cols_int = names + ["Total"]
    cols_float = ["Precision", "Recall", "F1-score"]

    idx = names + [name_last_row]
    df = pd.DataFrame(data=all_data, columns=cols_int + cols_float, index=idx)
    df[cols_int] = df[cols_int].astype(int)
    return df


def write_metrics(path_prediction, filename, df):
    filename = filename.split(".")[0]
    path_metrics = os.path.join(path_prediction, "metrics")
    os.makedirs(path_metrics, exist_ok=True)

    path_tex = os.path.join(path_metrics, f"{filename}.tex")
    path_txt = os.path.join(path_metrics, f"{filename}.txt")
    print(path_tex)

    # write tex file
    column_format = "|l|" + df.shape[1] * "r|"
    with open(path_tex, "w") as f:
        f.write(
            df.to_latex(
                bold_rows=True,
                float_format="{:0.2f}".format,
                column_format=column_format,
            )
        )

    with open(path_txt, "w") as f:
        df.to_string(f)
    print(f"* Metrics written to: {path_tex} and {path_tex}")


for path_ply in args.files:
    path_ply = os.path.join(args.prefix_path, path_ply)
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
    names = NAMES_9

    # in the 4-labels case
    if not config["data"]["all_labels"]:
        true_labels = convert_labels(true_labels).astype(np.int32)
        names = NAMES_4

    n = len(true_labels)
    predictions = -np.ones(n, dtype=np.int32)
    raw_predictions = predict(loader, len(dataset)).astype(np.int32)
    predictions[dataset.index] = raw_predictions
    errors = predictions != true_labels
    data["predictions"] = predictions
    data["errors"] = errors.astype(np.uint8)
    data["labels"] = true_labels

    # Save point cloud
    filename = os.path.basename(path_ply)
    path_prediction = os.path.join(ckpt_prediction_folder, filename)
    if dict2ply(data, path_prediction):
        print(f"* Predictions PLY file saved to: {path_prediction}")

    df = evaluate(
        true_labels[true_labels >= 0],
        raw_predictions,
        names,
        args.weighted_avg,
    )
    write_metrics(ckpt_prediction_folder, filename, df)
