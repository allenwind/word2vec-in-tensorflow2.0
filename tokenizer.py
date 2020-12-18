import json
import jieba
# from https://github.com/allenwind/count-in-parallel
from parallel import count_in_parallel_from_generator

jieba.initialize()

class Tokenizer:
    """支持并行的Tokenizer"""

    def __init__(self, mintf=16, processes=7):
        self.word2id = {}
        self.MASK = 0
        self.UNKNOW = 1
        self.mintf = mintf
        self.processes = processes
        self.filters = set("!\"#$%&'()[]*+,-./，。！@·……（）【】<>《》?？；‘’“”")

    def fit_in_parallel(self, X):
        tokenize = lambda x: jieba.lcut(x, HMM=False)
        words = count_in_parallel_from_generator(
            tokenize=tokenize,
            generator=X,
            processes=self.processes
        )

        # 过滤低频词和特殊符号
        words = {i: j for i, j in words.items() \
                 if j >= self.mintf and i not in self.filters}
        # 0:MASK
        # 1:UNK
        # 建立字词ID映射表
        for i, c in enumerate(words, start=2):
            self.word2id[c] = i
        self.words = words

    def __len__(self):
        return len(self.word2id) + 2

    def transform(self, X):
        wids = []
        for sentence in X:
            wid = self.transform_one(sentence)
            wids.append(wid)
        return wids

    def transform_one(self, sentence):
        wid = []
        for word in jieba.lcut(sentence, HMM=False):
            w = self.word2id.get(word, self.UNKNOW)
            wid.append(w)
        return wid

    def save(self, file):
        with open(file, "w") as fp:
            json.dump(
                [self.word2id, self.words],
                fp,
                indent=4,
                ensure_ascii=False
            )

    def load(self, file):
        with open(file, "r") as fp:
            self.word2id, self.words = json.load(fp)
