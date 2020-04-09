import torch
import torch.nn as nn
from torch.autograd import Variable

SIZE_RELATION_TYPE = {0: 0, 1: 1, 2: 3, 3: 4, 4: 3}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM(nn.Module):
    def __init__(
        self, n_features, num_classes, hidden_size, num_layers, relation_type=1
    ):
        super(BiLSTM, self).__init__()
        self.relation_type = relation_type

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        try:
            size_relation_vector = SIZE_RELATION_TYPE[relation_type]
        except KeyError:
            print(f"Relation type '{self.relation_type}' not recognized")
            return

        input_size = n_features + size_relation_vector
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x, debug=False):

        if debug:
            print("\ninput size")
            print(x.shape)
        x = self.relation_vectors(x)

        if debug:
            print("after transform")
            print(x.shape)

        batch_size = x.shape[0]
        hidden = self.init_hidden(batch_size)
        if debug:
            print("batch size")
            print(batch_size)
            print("hidden")
            print(hidden[0].shape)

        # Propagate input through LSTM :
        out, hidden = self.lstm(x, hidden)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        if debug:
            print("output size")
            print(out.shape)

        return out

    def init_hidden(self, batch_size):
        # initialization of hidden states
        # shape (num_layers * num_directions, batch, hidden_size)
        hidden = (
            Variable(
                torch.zeros(
                    self.num_layers * 2,
                    batch_size,
                    self.hidden_size,
                    device=device,
                )
            ),
            Variable(
                torch.zeros(
                    self.num_layers * 2,
                    batch_size,
                    self.hidden_size,
                    device=device,
                )
            ),
        )
        return hidden

    def relation_vectors(self, x):
        coords = x[:, :, :3]
        # shape : (batch_size, seq_len, 3)

        # no relation
        if self.relation_type == 0:
            return x[:, :, 3:]

        # distances only
        if self.relation_type == 1:
            diff = coords - coords[:, 0:1, :]
            distances = torch.sum(diff ** 2, dim=2, keepdim=True)
            return torch.cat((distances, x[:, :, 3:]), dim=2)

        # centered coords
        elif self.relation_type == 2:
            diff = coords - coords[:, 0:1, :]
            return torch.cat((diff, x[:, :, 3:]), dim=2)

        # centered coords + distances
        elif self.relation_type == 3:
            diff = coords - coords[:, 0:1, :]
            distances = torch.sum(diff ** 2, dim=2, keepdim=True)
            return torch.cat((diff, distances, x[:, :, 3:]), dim=2)

        # decentralized coords
        elif self.relation_type == 4:
            decentralized = coords - torch.min(coords, dim=0, keepdim=True)[0]
            return torch.cat((decentralized, x[:, :, 3:]), dim=2)
