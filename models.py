from keras.layers import *
from keras.layers.merge import *
from keras.layers.embeddings import *
from keras.models import *

class model(object):
    def __init__(self, padding_length, dict_length):
        self._dict_length = dict_length
        self._padding_length = padding_length

    @staticmethod
    def shared_SeNetwork(share_layers):

        def f(i):
            temp = None
            for j in share_layers:
                temp = j(i)
            return temp

        return f

    @staticmethod
    def shared_TextNetwork(share_layers):

        def f(i):
            temp = None
            for j in share_layers:
                temp = j(i)
            return temp

        return f

    @staticmethod
    def ShareLayerForSe():
        return [CuDNNGRU(64)]

    @staticmethod
    def ShareLayerForText():
        return [CuDNNGRU(64)]

    def SharedTextEmbedding(self, ShareLayerSe, ShareLayerText):
        dict_length = self._dict_length

        def f(i):
            embedding = Embedding(input_dim=dict_length, output_dim=50)(i)
            sentiment = self.shared_SeNetwork(ShareLayerSe)(embedding)
            TextLabel = self.shared_TextNetwork(ShareLayerText)(embedding)
            return Concatenate()([sentiment, TextLabel])

        return f

    def SharedNodeEmbedding(self, ShareLayerSe, ShareLayerText):
        def f(i, j):
            text_embedding = self.SharedTextEmbedding(ShareLayerSe, ShareLayerText)(j)
            return Concatenate()([i, text_embedding])

        return f

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
        time_input_a = Input(shape=(1,))
        text_input_a = Input(shape=(self._padding_length,))
        time_input_b = Input(shape=(1,))
        text_input_b = Input(shape=(self._padding_length,))
        node_embedding_a = self.NodeEmbedding()(time_input_a, text_input_a)
        node_embedding_b = self.NodeEmbedding()(time_input_b, text_input_b)

        y = [node_embedding_a, node_embedding_b]
        return Model(inputs=[time_input_a, text_input_a, time_input_b, text_input_b], outputs=y, name='Summary')

    def modelv2(self):
        time_input_a = Input(shape=(1,))
        text_input_a = Input(shape=(self._padding_length,))
        time_input_b = Input(shape=(1,))
        text_input_b = Input(shape=(self._padding_length,))
        SeLayer = self.ShareLayerForSe()
        TextLayer = self.ShareLayerForText()
        node_embedding_a = self.SharedNodeEmbedding(SeLayer, TextLayer)(time_input_a, text_input_a)
        node_embedding_b = self.SharedNodeEmbedding(SeLayer, TextLayer)(time_input_b, text_input_b)

        y = [node_embedding_a, node_embedding_b]
        return Model(inputs=[time_input_a, text_input_a, time_input_b, text_input_b], outputs=y, name='Summary')


    # def NodeEmbedding(self):
    #     time_input = Input(shape=(1, ))
    #     text_input = Input(shape=(self._padding_length,))
    #     text_embedding = self.TextEmbedding()(text_input)
    #     NodeEmbedding = Concatenate()([time_input, text_embedding])
    #
    #     return Model(inputs=[time_input, text_input], outputs=NodeEmbedding, name="NodeEmbedding")

if __name__ == '__main__':
    c = model(10, 1000)
    m1 = c.modelv2()
    print(m1.summary())
