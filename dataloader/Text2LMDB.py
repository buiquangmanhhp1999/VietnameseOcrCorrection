from tool.utils import extract_phrases, gen_ngrams
import itertools
from tqdm import tqdm
from config import NGRAM
import re
import lmdb
from unicodedata import normalize


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

# load data from file txt
label_file = 'train_data.txt'
lines = open(label_file, 'r').readlines()

cache = {}
error = 0
val_data = 6000000


# open lmdb database
train_env = lmdb.open('./train_lmdb', map_size=10099511627776)
val_env = lmdb.open('./val_lmdb', map_size=10099511627776)

phrases = itertools.chain.from_iterable(extract_phrases(text) for text in lines)
phrases = [p.strip() for p in phrases if len(p.split()) > 1]

char_regrex = '^[_aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 !\"\',\-\.:;?_\(\)]+$'
train_cnt = 0
val_cnt = 0

tgt_env = val_env
for p in tqdm(phrases, desc='Creating dataset ...'):
    if not re.match(char_regrex, p):
        continue

    for ngr in gen_ngrams(p, NGRAM):
        if len(" ".join(ngr)) < NGRAM * 7:
            ngram_text = " ".join(ngr)

            if val_cnt == val_data:
                if len(cache) > 0:
                    writeCache(tgt_env, cache)
                    cache = {}
                tgt_env = train_env

            if val_cnt < val_data:
                # write data
                textKey = 'text-%12d' % val_cnt
                val_cnt += 1
            else:
                # write data
                textKey = 'text-%12d' % train_cnt
                train_cnt += 1

            ngram_text = ngram_text.strip()
            ngram_text = ngram_text.rstrip()
            ngram_text = normalize("NFC", ngram_text)

            cache[textKey] = ngram_text.encode()
            if len(cache) % 1000 == 0:
                writeCache(tgt_env, cache)
                cache = {}

if len(cache) > 0:
    writeCache(tgt_env, cache)

cache = {}
cache['num-samples'] = str(train_cnt).encode()
writeCache(train_env, cache)

cache = {}
cache['num-samples'] = str(val_cnt).encode()
writeCache(val_env, cache)