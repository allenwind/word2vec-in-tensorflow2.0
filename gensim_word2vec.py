from gensim.models import Word2Vec
from gensim.test.utils import common_texts

# gensim==3.8.3

model = Word2Vec(sentences=common_texts, size=128, window=5, min_count=1, workers=4)
model.save("word2vec.model")
