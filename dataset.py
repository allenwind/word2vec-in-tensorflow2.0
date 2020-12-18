import glob
import os
import itertools
import random
import collections
import numpy as np
import jieba
import tensorflow as tf
from tokenizer import Tokenizer

window = 5 # 窗口大小
minlen = 30 # 句子最小长度
mintf = 64 # 最小词频
processes = 7 # 并行分词进程数

def preprocess(content):
    # 文章的预处理，这里暂不处理
    return content

_THUContent = "/home/zhiwen/workspace/dataset/THUCTC/THUCNews/**/*.txt"
def load_sentences(file=_THUContent, shuffle=True, limit=None):
    files = glob.glob(file)
    if shuffle:
        random.shuffle(files)
    for file in files[:limit]:
        with open(file, encoding="utf-8") as fd:
            content = fd.read()
        yield preprocess(content)

file = "word_meta.json"
tokenizer = Tokenizer(mintf, processes)
if os.path.exists(file):
    tokenizer.load(file)
else:
    X = load_sentences(limit=None)
    print("tokenize...")
    tokenizer.fit_in_parallel(X)
    tokenizer.save(file)

words = tokenizer.words
word2id = tokenizer.word2id
id2word = {j:i for i,j in word2id.items()}
vocab_size = len(tokenizer)

def create_subsamples(words, subsample_eps=1e-5):
    # 计算降采样表，用于context
    total = len(words)
    subsamples = {}
    for i, j in words.items():
        j = j / total
        if j <= subsample_eps:
            continue

        j = subsample_eps / j + (subsample_eps / j) ** 0.5
        if j >= 1.0:
            continue

        subsamples[word2id[i]] = j
    return subsamples

# 采样表
subsamples = create_subsamples(words)

def DataGenerator(epochs=10):
    for epoch in range(epochs):
        sentences = load_sentences()
        for sentence in sentences:
            # 关闭新词发现功能
            sentence = jieba.lcut(sentence, HMM=False)
            sentence = [0] * window + [word2id[w] for w in sentence if w in word2id] + [0] * window
            probs = np.random.random(len(sentence))
            for i in range(window, len(sentence) - window):
                # 满足降采样条件的直接跳过
                c = sentence[i]
                if c in subsamples and probs[i] > subsamples[c]:
                    continue
                # 窗口中的内容
                x = np.array(sentence[i-window:i] + sentence[i+1:i+window+1])
                c = np.array([c])
                # 为方便直接把target放在位置0
                z = np.array([0])
                yield (x, c), z

def create_dataset(window, minlen, batch_size=32, epochs=10):
    pass

dl = tf.data.Dataset.from_generator(
    DataGenerator,
    output_types=((tf.int32, tf.int32), tf.int32)
).shuffle(
    buffer_size=1024
).padded_batch(
    batch_size=320,
    padded_shapes=(([None], [None]), [None]),
    drop_remainder=True
).prefetch(tf.data.experimental.AUTOTUNE)

if __name__ == "__main__":
    # 测试
    for (a, b), c in iter(dl):
        print(a.shape, b.shape, c.shape)
        break
