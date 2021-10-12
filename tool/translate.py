import torch
import numpy as np
from torch.nn.functional import log_softmax, softmax

def translate(src_text, model, device, max_seq_length=256, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    model.eval()
    src_text = src_text.to(device)

    with torch.no_grad():
        memory = model.forward_encoder(src_text)
        translated_sentence = [[sos_token] * len(src_text)]
        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = model.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices = torch.topk(output, 5)
            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence