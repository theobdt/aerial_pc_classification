import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import math
import os
import matplotlib.pyplot as plt

from utils.dataloader import AerialPointDataset
from model import BiLSTM

PATH_FEATURES = "data/features"

# hyper-parameters
BATCH_SIZE = 1000
NUM_EPOCHS = 1
NUM_NEIGHBOORS = 20  # k of kNN
LEARNING_RATE = 0.001
NUM_LAYERS = 10  # nb of layers in lSTM network
HIDDEN_SIZE = 20  # nb of hidden units
INPUT_SIZE = 8  # nb of features
NUM_CLASSES = 9
NUM_WORKERS = 2  # nb of parallel workers


parser = argparse.ArgumentParser(description="Training")

parser.add_argument(
    "--file",
    "-f",
    type=str,
    nargs="+",
    default="vaihingen3D_test.ply",
    help="Path to the processed point cloud file",
)
parser.add_argument(
    "-bs", "--batch_size", type=int, default=BATCH_SIZE, help="Batch size"
)
parser.add_argument(
    "--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs"
)
parser.add_argument(
    "-k",
    "--kNN",
    type=int,
    default=NUM_NEIGHBOORS,
    help="Number of nearest neighboors",
)
parser.add_argument(
    "--lr", type=float, default=LEARNING_RATE, help="Start learning rate",
)
parser.add_argument(
    "--num_layer",
    type=int,
    default=NUM_LAYERS,
    help="Number of layers in lSTM network",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=HIDDEN_SIZE,
    help="Number of features in the hidden state",
)
parser.add_argument(
    "--input_size",
    type=int,
    default=INPUT_SIZE,
    help="Number of features in the input",
)
parser.add_argument(
    "--num_class", type=int, default=NUM_CLASSES, help="Number of classes",
)
parser.add_argument(
    "--num_workers", type=int, default=NUM_WORKERS, help="Number of workers",
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use : {device}")

print(f"\nTraining from file {args.file[0]}")
training_data = os.path.join(PATH_FEATURES, args.file[0])
dataset = AerialPointDataset(training_data, args.kNN)
train_loader = DataLoader(
    dataset=dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)

# Define model
model = BiLSTM(
    args.input_size, args.hidden_size, args.num_class, args.num_layer
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def decentralized_coordinate(coords):
    decentralized_coords = coords - torch.min(coords, axis=0).values
    return decentralized_coords


total_samples = len(dataset)
n_iter = math.ceil(total_samples / args.batch_size)
print(
    f"\nTotal samples : {total_samples} ; " f"Number of iterations : {n_iter}"
)

history_loss_train = []
history_acc_train = []
n_correct = 0
n_total = 0

# Training loop
for epoch in range(args.num_epochs):
    for i, (sequence, label) in enumerate(train_loader):

        # sequence shape = (batch_size, [point ; neighbors], nb_features)
        sequence[:, :, :3] = decentralized_coordinate(sequence[:, :, :3])
        label = label.type(dtype=torch.long)

        # Forward pass
        output = model(sequence, debug=False)
        train_loss = criterion(output, label)

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # calculate accuracy of predictions in the current batch
        n_correct = (torch.max(output, 1).indices == label).sum().item()
        n_total = BATCH_SIZE
        train_acc = 100.0 * n_correct / n_total

        history_loss_train.append(train_loss.item())
        history_acc_train.append(train_acc)

        if (i + 1) % 10 == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, "
                "Acc: {:.2f} %".format(
                    epoch + 1,
                    args.num_epochs,
                    i + 1,
                    n_iter,
                    train_loss.item(),
                    sum(history_acc_train[-10:]) / 10,
                )
            )

plt.figure()
plt.plot(history_loss_train)
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.figure()
plt.plot(history_acc_train)
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
