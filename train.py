import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import matplotlib.pyplot as plt
import yaml
from datetime import datetime
import shutil

from utils.dataloader import AerialPointDataset
from models import BiLSTM


parser = argparse.ArgumentParser(description="Training")
parser.add_argument(
    "--config",
    type=str,
    default="cfg/config_bilstm.yaml",
    help="Path to the config file",
)
parser.add_argument(
    "--debug", action="store_true", help="Debug mode",
)
parser.add_argument(
    "--log_interval", type=int, default=10, help="Log interval for training",
)
parser.add_argument(
    "--path_ckpts",
    type=str,
    default="ckpts",
    help="Path to the checkpoint folder",
)
parser.add_argument(
    "--prefix_path", type=str, default='', help="Path prefix",
)
parser.add_argument(
    "--resume", "-r", type=str, help="Name of the checkpoint to resume",
)
parser.add_argument(
    "--data_train",
    type=str,
    default="data/features/vaihingen3D_train.ply",
    help="Path to training data",
)
parser.add_argument(
    "--data_test",
    type=str,
    default="data/features/vaihingen3D_test.ply",
    help="Path to test data",
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use : {device}")


def init_ckpt():
    checkpoint = {
        "epoch": 0,
        "model_state_dict": None,
        "optimizer_state_dict": None,
        "losses": [],
        "accuracies": [],
        "best_train_loss": float("inf"),
    }
    return checkpoint


if args.resume:
    path_ckpt = os.path.join(args.prefix_path, args.path_ckpts, args.resume)
    print(f"Loading checkpoint {path_ckpt}")
    path_config = os.path.join(path_ckpt, "config.yaml")
    path_ckpt_dict = os.path.join(path_ckpt, "ckpt.pt")
    checkpoint = torch.load(path_ckpt_dict, map_location=device)
else:
    ckpt_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_ckpt = os.path.join(args.prefix_path, args.path_ckpts, ckpt_name)
    os.makedirs(path_ckpt, exist_ok=True)
    path_config = os.path.join(path_ckpt, "config.yaml")
    shutil.copy2(args.config, path_config)
    path_ckpt_dict = os.path.join(path_ckpt, "ckpt.pt")
    checkpoint = init_ckpt()
    print(f"Initialized checkpoint {path_ckpt}")

print(f"\nConfig file: {path_config}")

with open(path_config, "r") as f:
    config = yaml.safe_load(f)

# get number of epochs
epoch_start = checkpoint["epoch"]
epoch_end = config["training"].pop("epoch_end")
print(f"Epoch start: {epoch_start}, Epoch end: {epoch_end}")

max_batches_train = config["training"].pop("max_batches")
max_batches_test = config["test"].pop("max_batches")

path_train_ply = os.path.join(args.prefix_path, args.data_train)
path_test_ply = os.path.join(args.prefix_path, args.data_test)
print(f'\nTraining file: {path_train_ply}')
print(f'Test file: {path_test_ply}')

dataset_train = AerialPointDataset(path_train_ply, **config["data"])
dataset_test = AerialPointDataset(path_test_ply, **config["data"])

train_loader = DataLoader(dataset=dataset_train, **config["training"])
test_loader = DataLoader(dataset=dataset_test, **config["test"])


print(
    f"Total samples train: {len(dataset_train)}, "
    f"Number of batches: {len(train_loader)}"
    f"\nTotal samples test: {len(dataset_test)}, "
    f"Number of batches: {len(test_loader)}"
)

# Define model
n_features = len(config["data"]["features"])
n_classes = 4
if config["data"]["all_labels"]:
    n_classes = 9
print(f"Num classes: {n_classes}\n")
model = BiLSTM(n_features, n_classes, **config["network"]).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
lr = config["optimizer"]["learning_rate"]
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if args.resume:
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def decentralized_coordinate(coords):
    decentralized_coords = coords - torch.min(coords, axis=0).values
    return decentralized_coords


def train(loader, log_interval, max_batches=None):
    model.train()
    if max_batches is None:
        max_batches = len(loader)
    history_acc_train = []
    history_loss_train = []
    for i, (sequence, label) in enumerate(train_loader):
        sequence = sequence.to(device)
        label = label.to(device)

        label = label.type(dtype=torch.long)

        # Forward pass
        output = model(sequence, debug=args.debug)
        train_loss = criterion(output, label)

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # calculate accuracy of predictions in the current batch
        n_correct = (torch.max(output, 1).indices == label).sum().item()
        train_acc = 100.0 * n_correct / len(label)

        history_loss_train.append(train_loss.item())
        history_acc_train.append(train_acc)

        if (i + 1) % log_interval == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, "
                "Acc: {:.2f} %".format(
                    epoch + 1,
                    epoch_end,
                    i + 1,
                    max_batches,
                    train_loss.item(),
                    sum(history_acc_train[-log_interval:]) / log_interval,
                )
            )
        if i + 1 > max_batches:
            break
    return history_loss_train, history_acc_train


def evaluate(loader, max_batches=None):
    model.eval()
    if max_batches is None:
        max_batches = len(loader)
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for i, (sequence, label) in enumerate(loader):
            sequence = sequence.to(device)
            label = label.to(device)
            label = label.type(dtype=torch.long)
            output = model(sequence)

            total_loss += criterion(output, label)

            n_correct = (torch.max(output, 1).indices == label).sum().item()
            total_correct += n_correct / len(label)
            if i + 1 > max_batches:
                break
    avg_loss = total_loss / max_batches
    avg_acc = 100 * total_correct / max_batches
    print(f"Test Loss : {avg_loss:.4f}, Test Acc : {avg_acc:.2f} %\n")
    return avg_loss, avg_acc


# Training loop
for epoch in range(epoch_start, epoch_end):
    hist_loss_train, hist_acc_train = train(
        train_loader,
        log_interval=args.log_interval,
        max_batches=max_batches_train,
    )
    test_loss, test_acc = evaluate(test_loader, max_batches=max_batches_test)

    # update checkpoint
    checkpoint["losses"].append((hist_loss_train, test_loss))
    checkpoint["accuracies"].append((hist_acc_train, test_acc))

    avg_train_loss = sum(hist_loss_train) / len(hist_loss_train)
    if avg_train_loss < checkpoint["best_train_loss"]:
        checkpoint["best_train_loss"] = avg_train_loss
        checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path_ckpt_dict)


def plot_metric(ax, metric, label):
    train_values = []
    x_test = []
    y_test = []
    for (hist_train, test_value) in metric:
        train_values += hist_train
        x_test.append(len(train_values))
        y_test.append(test_value)
    ax.plot(train_values, label="train_" + label)
    ax.plot(x_test, y_test, label="test_" + label)
    ax.set_xlabel("Batches")
    ax.legend()


_, axes = plt.subplots(nrows=2, sharex=True)
axes[0].set_title("Loss")
plot_metric(axes[0], checkpoint["losses"], "loss")
axes[0].set_ylim([0, 2])

axes[1].set_title("Accuracy")
plot_metric(axes[1], checkpoint["accuracies"], "acc")
axes[1].set_ylim([0, 100])
plt.tight_layout()
path_fig = os.path.join(path_ckpt, "figure.png")
plt.savefig(path_fig)
print(f"Figure saved to {path_fig}")
plt.show()
