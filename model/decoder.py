import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import Attention

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = Attention(hidden_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size * 3, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size * 4, output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        inputs = inputs.unsqueeze(0)
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        
        # caculate attention weight
        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        
        # apply attention weight to encoder output
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        attn_applied = attn_applied.permute(1, 0, 2)
        
        # decoder
        rnn_input = torch.cat((embedded, attn_applied), 2)
        output, hidden = self.gru(rnn_input, hidden.unsqueeze(0))
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        attn_applied = attn_applied.squeeze(0)
        
        fc_input = torch.cat((output, attn_applied, embedded), dim=1)
        prediction = self.fc_out(fc_input)
        
        return prediction, hidden.squeeze(0), attn_weights