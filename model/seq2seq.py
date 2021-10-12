import torch
import torch.nn as nn
import torch.nn.functional as F
from model.decoder import AttnDecoderRNN
from model.encoder import Encoder

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, encoder_hidden, decoder_hidden, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, encoder_hidden, dropout)
        self.decoder = AttnDecoderRNN(decoder_hidden, vocab_size)

    def forward_encoder(self, src):
        """
        src: timestep x batch_size x channel
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """

        encoder_outputs, hidden = self.encoder(src)

        return hidden, encoder_outputs

    def forward_decoder(self, tgt, memory):
        """
        tgt: timestep x batch_size
        hidden: batch_size x hid_dim
        encoder: src_len x batch_size x hid_dim
        output: batch_size x 1 x vocab_size
        """

        tgt = tgt[-1]
        hidden, encoder_outputs = memory
        output, hidden, _ = self.decoder(tgt, hidden, encoder_outputs)
        output = output.unsqueeze(1)

        return output, (hidden, encoder_outputs)

    def forward(self, src, trg):
        """
        src: time_step x batch_size
        trg: time_step x batch_size
        outputs: batch_size x time_step x vocab_size
        """
        batch_size = src.shape[0]
        trg_len = src.shape[1]
        trg_vocab_size = self.decoder.output_size
        device = src.device

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        encoder_outputs, hidden = self.encoder(src)
        for t in range(trg_len):
            input = trg[t]
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output

        outputs = outputs.transpose(0, 1).contiguous()
        return outputs
