from keras.layers import *
from keras.layers.merge import *
from keras.layers.embeddings import *
from keras.models import *

class model(object):
    def __init__(self, padding_length, dict_length):
        self._dict_length = dict_length
        self._padding_length = padding_length

    @staticmethod
    def SeNetwork():

        def f(i):
            return CuDNNLSTM(64)(i)

        return f

    @staticmethod
    def TextNetwork():

        def f(i):
            return CuDNNLSTM(64)(i)

        return f

    def TextEmbedding(self):
        dict_length = self._dict_length

        def f(i):
            embedding = Embedding(input_dim=dict_length, output_dim=50)(i)
            sentiment = self.SeNetwork()(embedding)
            TextLabel = self.TextNetwork()(embedding)
            return Concatenate()([sentiment, TextLabel])

        return f

    def NodeEmbedding(self):
        def f(i, j):
            text_embedding = self.TextEmbedding()(j)
            return Concatenate()([i, text_embedding])

        return f

    def modelv1(self):
        time_input = Input(shape=(1,))
        text_input = Input(shape=(self._padding_length,))
        text_embedding = self.TextEmbedding()(text_input)
        node_embedding = self.NodeEmbedding()(time_input, text_embedding)

        y = node_embedding
        return Model(inputs=[time_input, text_input], outputs=y, name='Summary')

    # def NodeEmbedding(self):
    #     time_input = Input(shape=(1, ))
    #     text_input = Input(shape=(self._padding_length,))
    #     text_embedding = self.TextEmbedding()(text_input)
    #     NodeEmbedding = Concatenate()([time_input, text_embedding])
    #
    #     return Model(inputs=[time_input, text_input], outputs=NodeEmbedding, name="NodeEmbedding")

if __name__ == '__main__':
    c = model(10, 1000)
    m1 = c.modelv1()
    print(m1.summary())
