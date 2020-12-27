import pprint
import numpy as np
from model_cbow import model as cbow
from model_skipgram import model as skipgram
from dataset import word2id, id2word

MODEL = "cbow" # skipgram

# embeddings权重
if MODEL == "cbow":
    embeddings = cbow.get_weights()[0]
else:
    embeddings = skipgram.get_weights()[0]

def norm(x):
    x = x / np.sqrt(np.square(x).sum(axis=1).reshape((-1, 1)))
    return x

embeddings = norm(embeddings)

def topk_similar(word, topk=10):
    wid = word2id.get(word, None)
    if wid is None:
        raise KeyError("word '{}' not in word tables".format(word))
    vector = embeddings[wid]
    similars = np.dot(embeddings, vector)
    r = similars.argsort()[::-1]
    r = r[r > 0]
    return [(id2word[i], similars[i]) for i in r[:topk]]

if __name__ == "__main__":
    pprint.pprint(topk_similar("发展"))
    pprint.pprint(topk_similar("中国"))
    pprint.pprint(topk_similar("美国"))
    pprint.pprint(topk_similar("股票"))
    pprint.pprint(topk_similar("资本"))
    pprint.pprint(topk_similar("苹果"))
    pprint.pprint(topk_similar("数学"))
    pprint.pprint(topk_similar("算法"))
    pprint.pprint(topk_similar("NVIDIA"))
