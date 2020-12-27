# word2vec-in-tensorflow2.0

基于Tensorflow2.x的Word2vec。Word2vec是求取词向量的工具，包括两个模型（CBOW和SkipGram）和两种优化方案（Hierarchical softmax和负采样），这里开源负采样、CBOW和SkipGram的实现。


## CBOW

模型训练（需要训练语料，可自行调整）：

```bash
$ python model_cbow.py
```

模型训练好后简单的相似计算：

```bash
$ python tooks.py
```

简单的交互测试：

```bash
>>> from tools import topk_similar
>>> import pprint
>>> pprint.pprint(topk_similar("数学"))
[('数学', 1.0000001),
 ('语文', 0.9285573),
 ('试题', 0.9178058),
 ('英语', 0.91363966),
 ('专业课', 0.9079031),
 ('考', 0.9042375),
 ('高等数学', 0.897079),
 ('分值', 0.89574695),
 ('雅思', 0.893947),
 ('外语', 0.88838947)]

>>> pprint.pprint(topk_similar("资本"))
[('资本', 1.0),
 ('外资', 0.84800375),
 ('融资', 0.8301372),
 ('资产', 0.81580234),
 ('投资人', 0.78543866),
 ('投资', 0.7799219),
 ('并购', 0.77950025),
 ('承销', 0.7720045),
 ('发债', 0.76935923),
 ('管理层', 0.7664337)]

>>> pprint.pprint(topk_similar("CPU"))
[('CPU', 1.0),
 ('GPU', 0.9385017),
 ('1GHz', 0.9292523),
 ('功耗', 0.9110091),
 ('ROM', 0.90495074),
 ('2G', 0.904893),
 ('芯片', 0.90474397),
 ('芯', 0.90301657),
 ('低功耗', 0.8979711),
 ('256MB', 0.89573085)]
```

以上的训练直接使用[THUCNews语料](http://thuctc.thunlp.org/)。

## SkipGram


模型训练（需要训练语料，可自行调整）：

```bash
$ python model_skipgram.py
```

`tooks.py`中的参数调整为`skipgram`后做简单的相似计算：

```bash
$ python tooks.py
```

## 参考

[1] http://thuctc.thunlp.org/
