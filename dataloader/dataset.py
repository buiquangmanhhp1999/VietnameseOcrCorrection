from torch.utils.data import Dataset
import unidecode
from tqdm import tqdm
from torch.utils.data.sampler import Sampler
from collections import defaultdict
from tool.utils import get_bucket
import re
from config import alphabet, chars_regrex, same_chars
import torch
import numpy as np
from model.vocab import Vocab
import lmdb
import os
import json
import random


class BasicDataset(Dataset):
    def __init__(self, lmdb_path):
        # read data
        self.vocab = Vocab(alphabet)
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, max_readers=126, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)
        self.nSamples = int(self.txn.get('num-samples'.encode()))
        self.cluster_indices = defaultdict(list)
        self.build_cluster_indices()
    
    def build_cluster_indices(self):
        error = 0
        if not os.path.isfile(os.path.basename(self.lmdb_path) + '.json'):
            pbar = tqdm(range(self.__len__()), desc='{} build cluster'.format(self.__len__()), ncols=100, position=0, leave=True)
            for i in pbar:
                text = self.read_data(i)
                if text is None or len(text) > 200:
                    error += 1
                    continue

                w = len(text)
                bucket_size = get_bucket(w)
                self.cluster_indices[int(bucket_size)].append(i)

            print('Remove {} images'.format(error))
            data = json.dumps(self.cluster_indices)
            f = open(os.path.basename(self.lmdb_path) + '.json', "w")
            f.write(data)
            f.close()
        else:
            # Opening JSON file
            f = open(os.path.basename(self.lmdb_path) + '.json',)

            # returns JSON object as
            # a dictionary
            data = json.load(f)

            for bucket_size in data:
                if len(data[str(bucket_size)]) > 0:
                    self.cluster_indices[int(bucket_size)].extend(data[str(bucket_size)])

    def read_data(self, idx):
        textKey = 'text-%12d' % idx
        text = self.txn.get(textKey.encode()).decode()

        return text

    def remove_random_accent(self, text, ratio=0.15):
        words = text.split()
        mask = np.random.random(size=len(words)) < ratio

        for i in range(len(words)):
            if mask[i]:
                words[i] = unidecode.unidecode(words[i])
                break

        return ' '.join(words)

    def remove_random_space(self, text):
        words = text.split()
        n_words = len(words)
        start = np.random.randint(low=0, high=n_words, size=1)[0]

        if start + 3 < n_words:
            end = np.random.randint(low=start, high=start + 3, size=1)[0]
        else:
            end = np.random.randint(low=start, high=n_words, size=1)[0]

        out = ' '.join(words[:start])  + ' ' + ''.join(words[start:end + 1]) + ' ' + ' '.join(words[end + 1:])

        return out.strip()

    def _char_regrex(self, text):
        match_chars = re.findall(chars_regrex, text)

        return match_chars

    def _random_replace(self, text, match_chars):
        replace_char = match_chars[np.random.randint(low=0, high=len(match_chars), size=1)[0]]
        insert_chars = same_chars[unidecode.unidecode(replace_char)]
        insert_char = insert_chars[np.random.randint(low=0, high=len(insert_chars), size=1)[0]]
        text = text.replace(replace_char, insert_char, 1)

        return text

    def change(self, text):
        match_chars = self._char_regrex(text)
        if len(match_chars) == 0:
            return text

        text = self._random_replace(text, match_chars)

        return text

    def replace_accent_chars(self, text, ratio=0.15):
        words = text.split()
        mask = np.random.random(size=len(words)) < ratio

        for i in range(len(words)):
            if mask[i]:
                words[i] = self.change(words[i])
                break

        return ' '.join(words)

    def upper(self, text, ratio=0.1):
        words = text.split()
        mask_words = np.random.random(size=len(words)) < ratio
        cnt_word = 0

        for i in range(len(words)):
            if mask_words[i] and len(words[i]) > 1:
                mask_chars = np.random.random(size=len(words[i])) < ratio
                cnt_char = 0
                for j in range(1, len(words[i])):
                    if mask_chars[j]:
                        words[i] = words[i].replace(words[i][j], words[i][j].upper())
                        cnt_char += 1

                    if cnt_char >= 3:
                        break

                cnt_word += 1

            if cnt_word >= 1:
                break

        return  ' '.join(words)

    def __getitem__(self, idx):
        ori = self.read_data(idx)
        gts = ori

        if len(ori.split()) > 3:
            rd_aug_idx = np.random.randint(low=0, high=9, size=1)[0]
            if rd_aug_idx < 3:
                text = self.remove_random_accent(ori, ratio=0.3)
            elif rd_aug_idx < 6:
                text = self.replace_accent_chars(ori, ratio=0.3)
            elif rd_aug_idx < 8:
                text  = self.upper(ori, ratio=0.3)
            else:
                text = self.remove_random_space(ori)
        else:
            text = self.remove_random_space(ori)

        input_text = self.vocab.encode(text)
        gts = self.vocab.encode(gts)

        return {'text': input_text, 'label': gts}

    def __len__(self):
        return self.nSamples

class ClusterRandomSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    @staticmethod
    def flatten_list(lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        for cluster, cluster_indices in self.data_source.cluster_indices.items():
            if self.shuffle:
                random.shuffle(cluster_indices)

            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]

            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        batch_lists = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(batch_lists)

        batch_lists = self.flatten_list(batch_lists)
        return iter(batch_lists)

    def __len__(self):
        return len(self.data_source)

class Collator(object):
    def __call__(self, batch):
        text_data = []
        tgt_input = []
        target_weights = []

        MAXLEN = max(len(sample['label']) for sample in batch)
        for sample in batch:
            text, label = sample['text'], sample['label']
            label_len = len(label)
            text_len = len(text)

            src = np.concatenate((text, np.zeros(MAXLEN - text_len, dtype=np.int32)))
            tgt = np.concatenate((label, np.zeros(MAXLEN - label_len, dtype=np.int32)))

            text_data.append(src)
            tgt_input.append(tgt)
            one_mask_len = label_len - 1

            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(MAXLEN - one_mask_len, dtype=np.float32))))

        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1] = 0
        tgt_padding_mask = np.array(target_weights) == 0


        return torch.LongTensor(text_data), torch.LongTensor(tgt_input), torch.LongTensor(tgt_output), torch.BoolTensor(tgt_padding_mask)