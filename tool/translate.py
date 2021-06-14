import torch
import numpy as np 
from config import MAXLEN


def translate(src_text, model, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    model.eval()
    device = src_text.device

    with torch.no_grad():
        memory = model.forward_encoder(src_text)
        translated_sentence = [[sos_token] * len(src_text)]
        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = model.forward_decoder(tgt_inp, memory)
            output = output.to('cpu')
            
            values, indices = torch.topk(output, 5)
            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence


def preprocess(vocab, text):
    if isinstance(text, str):
        text = vocab.encode(text)
        text = np.expand_dims(text, axis=0)
        return torch.LongTensor(text)
    elif isinstance(text, list):
        src_text = []
        max_len = MAXLEN
        
        for txt in text:
            txt = vocab.encode(txt)
            text_len = len(txt)
            src = np.concatenate((txt, np.zeros(MAXLEN - text_len, dtype=np.int32)))
            src_text.append(src)
            
        return torch.LongTensor(src_text)
            
            
            