import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from dataset import dl, dl_sg, vocab_size

class SkipGram(tf.keras.layers.Layer):

    def __init__(self, input_dim, output_dim, window, **kwargs):
        super(SkipGram, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.span = tf.cast(2 * window, tf.float32)
        self.embedding = Embedding(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer="uniform"
        )
        self.dense = Dense(input_dim)

    def call(self, inputs):
        context, target = inputs
        hidden = self.embedding(context)
        hidden = tf.squeeze(hidden, axis=1)
        scores = self.dense(hidden) # (batch_size, vocab_size)
        # like tf.batch_gather in tf1.x
        target_scores = tf.gather(scores, target, batch_dims=-1)
        # SkipGram的目标是模型的多个输出的联合概率最大化
        loss = self.span * tf.reduce_logsumexp(scores, axis=1) - tf.reduce_sum(target_scores, axis=1)
        self.add_loss(tf.reduce_mean(loss))
        return loss


# 词向量的大小
output_dim = 128
# 单边窗口大小
window = 5
span = window * 2
input_dim = vocab_size

context = Input(shape=(1,), dtype=tf.int32)
target = Input(shape=(span,), dtype=tf.int32)
loss = SkipGram(input_dim, output_dim, window)([context, target])
model = Model(inputs=[context, target], outputs=loss)
model.summary()

model.compile(
    optimizer="adam",
)

if __name__ == "__main__":
    model.fit(
        dl_sg,
        steps_per_epoch=10000,
        epochs=5
    )
    model.save_weights("skipgram.weights")
else:
    model.load_weights("skipgram.weights")
