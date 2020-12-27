import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from dataset import dl, vocab_size

class CBOWLayer(tf.keras.layers.Layer):
    """计算context向量"""

    def __init__(self, input_dim, output_dim, **kwargs):
        super(CBOWLayer, self).__init__(**kwargs)
        self.embedding = Embedding(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer="uniform"
        )

    def call(self, inputs, mask=None):
        ws = self.embedding(inputs)
        # 这里直接使用平均做上下文向量，可以尝试其他的聚合方法
        context = tf.reduce_mean(ws, axis=1)
        return context

class NegativeSamplingLayer(tf.keras.layers.Layer):
    """随机构造负样本并与正样本组合"""

    def __init__(self, negative_count, vocab_size, **kwargs):
        super(NegativeSamplingLayer, self).__init__(**kwargs)
        self.negative_count = negative_count # 负样本数量
        self.vocab_size = vocab_size

    def call(self, inputs):
        x = inputs
        batch_size = tf.shape(x)[0]
        neg_samples = tf.random.uniform(
            shape=(batch_size, self.negative_count),
            minval=0,
            maxval=self.vocab_size,
            dtype=tf.int32
        )
        # (batch_size, negative_count + 1)
        # 把正样本放在位置0，需要与data pipeline配合
        x = tf.concat([x, neg_samples], axis=-1)
        return x

class RandomSoftmax(tf.keras.layers.Layer):
    """负采样优化计算softmax，减小计算量"""

    def __init__(self, input_dim, output_dim, **kwargs):
        super(RandomSoftmax, self).__init__(**kwargs)
        self.w_dense = Embedding(input_dim, output_dim)
        self.b_dense = Embedding(input_dim, 1)

    def call(self, inputs, mask=None):
        samples, context = inputs
        weights = self.w_dense(samples) # (batch_size, samples_size, output_dim)
        biases = self.b_dense(samples) # (batch_size, samples_size, 1)
        context = tf.expand_dims(context, axis=2) # # (batch_size, output_dim, 1)
        # batch内的矩阵乘积
        x = tf.einsum("aij,ajk->aik", weights, context) + biases # (batch_size, samples_size, 1)
        x = tf.squeeze(x, axis=2)
        x = tf.math.softmax(x, axis=-1) # (batch_size, samples_size)
        return x

# 词向量的大小
output_dim = 128
# 负样本数量
negative_count = 16
# 单边窗口大小
window = 5
span = window * 2
input_dim = vocab_size

words = Input(shape=(span,), dtype=tf.int32) # 滑动窗口内的词序列
target = Input(shape=(1,), dtype=tf.int32) # 目标词
context = CBOWLayer(input_dim, output_dim)(words) # # 上下文向量
samples = NegativeSamplingLayer(negative_count, input_dim)(target)
outputs = RandomSoftmax(input_dim, output_dim)([samples, context])
model = Model(inputs=[words, target], outputs=outputs)
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

if __name__ == "__main__":
    model.fit(
        dl,
        steps_per_epoch=10000,
        epochs=5
    )
    model.save_weights("word2vec.weights")
else:
    model.load_weights("word2vec.weights")
