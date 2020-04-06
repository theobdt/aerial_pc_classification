import torch
import torch.nn as nn
from torch.autograd import Variable


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=200):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
            print("input size")
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
                torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
            ),
            Variable(
                torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
            ),
        )
        return hidden
