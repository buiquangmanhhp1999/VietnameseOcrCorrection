from torch.utils.data import Dataset
import unidecode
import itertools
from tqdm import tqdm
from utils import extract_phrases, gen_ngrams
import re
from config import NGRAM, alphabet, MAXLEN
import torch
import numpy as np
from model.vocab import Vocab

class BasicDataset(Dataset):
    def __init__(self, label_file="./train_data.txt"):
        # read data 
        self.list_ngrams = self.load_data(label_file)
        self.vocab = Vocab(alphabet)
#         self.env = lmdb.open(lmdb_path, max_readers=126, readonly=True, lock=False, readahead=False, meminit=False)
#         self.txn = self.env.begin(write=False)
    
    def load_data(self, label_file):
        lines = open(label_file, 'r').readlines()
        phrases = itertools.chain.from_iterable(extract_phrases(text) for text in lines)
        phrases = [p.strip() for p in phrases if len(p.split()) > 1]
        
        list_ngrams = []
        char_regrex = '^[_aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 !\"\',\-\.:;?_\(\)]+$'
        
        for p in tqdm(phrases):
            if not re.match(char_regrex, p):
                continue
                
            if len(list_ngrams) > 100000:
                break
            for ngr in gen_ngrams(p, NGRAM):
                if len(" ".join(ngr)) < NGRAM * 7:
                    ngram_text = " ".join(ngr)
                    
                    list_ngrams.append(" ".join(ngr))
        list_ngrams = list(set(list_ngrams))
   
        return list_ngrams
    
    def read_data(self, idx):
        textKey = 'text-%12d' % idx
        text = self.txn.get(textKey.encode()).decode()
     
        return text
        
    
    def remove_random_accent(self, text):
        words = text.split()
        mask = np.random.random(size=len(words)) < 0.5

        for i in range(len(words)):
            if mask[i]:
                words[i] = unidecode.unidecode(words[i])

        return ' '.join(words)

#     def __getitem__(self, idx):
#         text = self.read_data(idx)
#         text = self.remove_random_accent(text)
#         gts = self.list_ngrams[idx]
        
#         input_text = self.vocab.encode(text)
#         label = self.vocab.encode(gts)
        
#         return {'text': input_text, 'word': label}
    
    def __getitem__(self, idx):
        text = unidecode.unidecode(self.list_ngrams[idx])
        gts = self.list_ngrams[idx]
        
        input_text = self.vocab.encode(text)
        label = self.vocab.encode(gts)
        
        return {'text': input_text, 'word': label}

    def __len__(self):
        return len(self.list_ngrams)


class Collator(object):
    def __call__(self, batch):
        text_data = []
        tgt_input = []
        
        for sample in batch:
            text, label = sample['text'], sample['word']  
            label_len = len(label)
            text_len = len(text)
            
            src = np.concatenate((text, np.zeros(MAXLEN - text_len, dtype=np.int32)))
            tgt = np.concatenate((label, np.zeros(MAXLEN - label_len, dtype=np.int32)))
            text_data.append(src)
            tgt_input.append(tgt)
        
        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1] = 0

        return torch.LongTensor(text_data), torch.LongTensor(tgt_input), torch.LongTensor(tgt_output)