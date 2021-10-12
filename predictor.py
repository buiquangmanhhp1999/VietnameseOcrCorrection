from model.seq2seq import Seq2Seq
from model.transformer import LanguageTransformer
import torch
from tool.translate import translate
from model.vocab import Vocab
from config import alphabet
import time
import numpy as np
import re
from collections import defaultdict
from utils import get_bucket
import unidecode
import time


class Predictor(object):
    def __init__(self, device, model_type='seq2seq', weight_path='./weights/seq2seq_0.pth'):
        if model_type == 'seq2seq':
            self.model = Seq2Seq(len(alphabet), encoder_hidden=256, decoder_hidden=256)
        elif model_type == 'transformer':
            self.model = LanguageTransformer(len(alphabet), d_model=256, nhead=4, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=768, max_seq_length=256, pos_dropout=0.1, trans_dropout=0.1)
        self.device = device
        self.model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        self.model = self.model.to(device)
        self.vocab = Vocab(alphabet)
        
    def preprocess(self, text):
        if isinstance(text, str):
            text = self.vocab.encode(text)
            text = np.expand_dims(text, axis=0)
            return torch.LongTensor(text)
        elif isinstance(text, list):
            src_text = []
            MAXLEN = max(len(txt) for txt in text) + 2

            for txt in text:
                txt = self.vocab.encode(txt)
                text_len = len(txt)
                src = np.concatenate((txt, np.zeros(MAXLEN - text_len, dtype=np.int32)))
                src_text.append(src)
            
        return torch.LongTensor(src_text).to(self.device)
    
    def extract_phrase(self, paragraph):
        # extract phrase
        return re.findall(r'\w[\w ]*|\s\W+|\W+', paragraph)
        
        
    def process(self, paragraph, NGRAM):
        phrases = self.extract_phrase(paragraph)
        inputs = []
        masks = []
        
        # group by n-grams
        for phrase in phrases:
            words = phrase.split()

            if len(words) < 2 or not re.match("\w[\w ]+", phrase):
                inputs.append(phrase)
                masks.append(False)
            else:
                for i in range(0, len(words), NGRAM):
                    inputs.append(' '.join(words[i:i + NGRAM]))
                    masks.append(True)
                    if len(words) - i - NGRAM < NGRAM:
                        inputs[-1] += ' ' + ' '.join(words[i + NGRAM:])
                        inputs[-1] = inputs[-1].strip()
                        break
        
        return inputs, masks
    
    def predict(self, paragraph, NGRAM=5):
        inputs, masks = self.process(paragraph, NGRAM)

        # preprocess and translate
        model_input = self.preprocess(list(np.array(inputs)[masks]))
        model_output = self._predict(model_input)
  
        results = ""
        idx = 0
        for i, mask in enumerate(masks):
            if mask:
                results += " " + model_output[idx]
                idx += 1
            else:
                results += inputs[i].strip()

        return results.strip()
    
    def _predict(self, model_input):
        model_output = translate(model_input, self.model, self.device).tolist()
        model_output = self.vocab.batch_decode(model_output)
        
        return model_output
    
    def batch_process(self, paragraphs, NGRAM):
        inputs = []
        masks = []
        para_len = []
        
        for p in paragraphs:
            phrases = self.extract_phrase(p)
            cnt = 0
            
            for phrase in phrases:
                words = phrase.split()
                
                if len(words) < 2 or not re.match("\w[\w ]+", phrase):
                    inputs.append(phrase.strip())
                    masks.append(False)
                    cnt += 1
                else:
                    for i in range(0, len(words), NGRAM):
                        inputs.append(' '.join(words[i:i + NGRAM]))
                        masks.append(True)
                        cnt += 1
                        
                        if len(words) - i - NGRAM < NGRAM:
                            inputs[-1] += ' ' + ' '.join(words[i + NGRAM:])
                            inputs[-1] = inputs[-1].strip()
                            break
                            
            para_len.append(cnt)
       
        return inputs, masks, para_len
    
    def batch_predict(self, paragraphs, NGRAM=5, batch_size=256):
        inputs, masks, para_len = self.batch_process(paragraphs, NGRAM)
        outputs = list()
        
        # build cluster 
        print(list(np.array(inputs)[masks]))
        cluster_texts, indices = self.build_cluster_texts(list(np.array(inputs)[masks]))
        
        # preprocess and translate
        for _, batch_texts in cluster_texts.items():
            if len(batch_texts) <= batch_size:
                model_input = self.preprocess(batch_texts)
                model_output = self._predict(model_input)
                outputs.extend(model_output)
            else:
                for i in range(0, len(batch_texts), batch_size):
                    model_input = self.preprocess(batch_texts[i:i + batch_size])
                    model_output = self._predict(model_input)
                    outputs.extend(model_output)
                    
        # sort result correspond to indices
        z = zip(outputs, indices)
        outputs = sorted(z, key=lambda x: x[1])
        outputs, _ = zip(*outputs)
        print('-----------------')
        print(outputs)
        
        # group n-grams -> final paragraphs
        para_idx = 0
        sentence_idx = 0
        paragraphs = []
        p = ""
        for i, mask in enumerate(masks):
            if para_len[para_idx] == i:
                paragraphs.append(p.strip())
                p = ""
                para_idx += 1
                
            if mask:
                p += " " + outputs[sentence_idx]
                sentence_idx += 1
            else:
                p += inputs[i].strip()
        
        return paragraphs        

    @staticmethod
    def sort_width(texts):
        batch = list(zip(texts, range(len(texts))))
        sorted_texts = sorted(batch, key=len, reverse=False)
        sorted_texts, indices = list(zip(*sorted_texts))

        return sorted_texts, indices

    def build_cluster_texts(self, texts):
        cluster_texts = defaultdict(list)
        sorted_texts, indices = self.sort_width(texts)

        for text in sorted_texts:
            cluster_texts[get_bucket(len(text))].append(text)

        return cluster_texts, indices
