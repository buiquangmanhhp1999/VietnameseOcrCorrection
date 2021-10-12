import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()

        self.rnn = nn.GRU(hidden_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)

    def forward(self, src):
        """
        src: batch_size x time_step x number_class
        outputs: batch_size x max_length x enc_hid_size ** 2
        hidden: batch_size x hid_dim
        """
        embedded = self.embedding(src)
        embedded = embedded.permute(1, 0, 2)
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden