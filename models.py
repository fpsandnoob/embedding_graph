# from keras.layers import *
# from keras.layers.merge import *
# from keras.layers.embeddings import *
# from keras.models import *
import tensorflow as tf
# from tensorflow.contrib.cudnn_rnn import CudnnGRU

import config

# class model_keras(object):
#     def __init__(self, padding_length, dict_length):
#         self._dict_length = dict_length
#         self._padding_length = padding_length
#
#     @staticmethod
#     def shared_SeNetwork(share_layers):
#
#         def f(i):
#             temp = None
#             for j in share_layers:
#                 temp = j(i)
#             return temp
#
#         return f
#
#     @staticmethod
#     def shared_TextNetwork(share_layers):
#
#         def f(i):
#             temp = None
#             for j in share_layers:
#                 temp = j(i)
#             return temp
#
#         return f
#
#     @staticmethod
#     def ShareLayerForSe():
#         return [CuDNNGRU(64)]
#
#     @staticmethod
#     def ShareLayerForText():
#         return [CuDNNGRU(64)]
#
#     def SharedTextEmbedding(self, ShareLayerSe, ShareLayerText):
#         dict_length = self._dict_length
#
#         def f(i):
#             embedding = Embedding(input_dim=dict_length, output_dim=50)(i)
#             sentiment = self.shared_SeNetwork(ShareLayerSe)(embedding)
#             TextLabel = self.shared_TextNetwork(ShareLayerText)(embedding)
#             return Concatenate()([sentiment, TextLabel])
#
#         return f
#
#     def SharedNodeEmbedding(self, ShareLayerSe, ShareLayerText):
#         def f(i, j):
#             text_embedding = self.SharedTextEmbedding(ShareLayerSe, ShareLayerText)(j)
#             return Concatenate()([i, text_embedding])
#
#         return f
#
#     @staticmethod
#     def SeNetwork():
#
#         def f(i):
#             return CuDNNLSTM(64)(i)
#
#         return f
#
#     @staticmethod
#     def TextNetwork():
#
#         def f(i):
#             return CuDNNLSTM(64)(i)
#
#         return f
#
#     def TextEmbedding(self):
#         dict_length = self._dict_length
#
#         def f(i):
#             embedding = Embedding(input_dim=dict_length, output_dim=50)(i)
#             sentiment = self.SeNetwork()(embedding)
#             TextLabel = self.TextNetwork()(embedding)
#             return Concatenate()([sentiment, TextLabel])
#
#         return f
#
#     def NodeEmbedding(self):
#         def f(i, j):
#             text_embedding = self.TextEmbedding()(j)
#             return Concatenate()([i, text_embedding])
#
#         return f
#
#     def Modelv1(self):
#         time_input_a = Input(shape=(1,))
#         text_input_a = Input(shape=(self._padding_length,))
#         time_input_b = Input(shape=(1,))
#         text_input_b = Input(shape=(self._padding_length,))
#         node_embedding_a = self.NodeEmbedding()(time_input_a, text_input_a)
#         node_embedding_b = self.NodeEmbedding()(time_input_b, text_input_b)
#
#         y = [node_embedding_a, node_embedding_b]
#         return Model(inputs=[time_input_a, text_input_a, time_input_b, text_input_b], outputs=y, name='Summary')
#
#     def Modelv2(self):
#         time_input_a = Input(shape=(1,))
#         text_input_a = Input(shape=(self._padding_length,))
#         time_input_b = Input(shape=(1,))
#         text_input_b = Input(shape=(self._padding_length,))
#         SeLayer = self.ShareLayerForSe()
#         TextLayer = self.ShareLayerForText()
#         node_embedding_a = self.SharedNodeEmbedding(SeLayer, TextLayer)(time_input_a, text_input_a)
#         node_embedding_b = self.SharedNodeEmbedding(SeLayer, TextLayer)(time_input_b, text_input_b)
#
#         y = [node_embedding_a, node_embedding_b]
#         return Model(inputs=[time_input_a, text_input_a, time_input_b, text_input_b], outputs=y, name='Summary')
#
#
#     # def NodeEmbedding(self):
#     #     time_input = Input(shape=(1, ))
#     #     text_input = Input(shape=(self._padding_length,))
#     #     text_embedding = self.TextEmbedding()(text_input)
#     #     NodeEmbedding = Concatenate()([time_input, text_embedding])
#     #
#     #     return Model(inputs=[time_input, text_input], outputs=NodeEmbedding, name="NodeEmbedding")


class model_tf(object):
    def __init__(self, vocab_size, num_nodes):
        with tf.name_scope('read_inputs') as scope:
            self.Text_a = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Text_a')
            self.Text_b = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Text_b')
            self.Node_a = tf.placeholder(tf.int32, [config.batch_size], name='node_1')
            self.Node_b = tf.placeholder(tf.int32, [config.batch_size], name='node_2')
            # self.Time_a = tf.placeholder(tf.int32, [config.batch_size], name='time_1')
            # self.Time_b = tf.placeholder(tf.int32, [config.batch_size], name='time_2')
            self.Polarity_a = tf.placeholder(tf.int32, [config.batch_size, 2], name='polarity_a')
            self.Polarity_b = tf.placeholder(tf.int32, [config.batch_size, 2], name='polarity_b')

        with tf.name_scope('initialize_embeddings') as scope:
            self.text_embed = tf.Variable(tf.truncated_normal([vocab_size, int(config.embed_size / 4)], stddev=0.3))
            self.node_embed = tf.Variable(tf.truncated_normal([vocab_size, int(config.embed_size / 4)], stddev=0.3))
            # self.time_embed = tf.Variable(tf.truncated_normal([vocab_size, int(config.embed_size / 4)], stddev=0.3))
            self.polarity_embed = tf.Variable(tf.truncated_normal([vocab_size, int(config.embed_size / 4)], stddev=0.3))
            self.node_embed = tf.clip_by_norm(self.node_embed, clip_norm=1, axes=1)

        with tf.name_scope('lookup_embeddings') as scope:
            self.T_A = tf.nn.embedding_lookup(self.text_embed, self.Text_a)
            # self.T_A = tf.expand_dims(self.TA, -1)

            self.T_B = tf.nn.embedding_lookup(self.text_embed, self.Text_b)
            # self.T_B = tf.expand_dims(self.TB, -1)

            self.N_A = tf.nn.embedding_lookup(self.node_embed, self.Node_a)
            self.N_B = tf.nn.embedding_lookup(self.node_embed, self.Node_b)

            # self.Time_A = tf.nn.embedding_lookup(self.time_embed, self.Time_a)
            self.Polarity_A = tf.nn.embedding_lookup(self.polarity_embed, self.Polarity_a)
            self.Polarity_B = tf.nn.embedding_lookup(self.polarity_embed, self.Polarity_b)

        self.gruA, self.gruB, self.resA, self.resB = self.TopicNetwork()
        self.loss = self.compute_loss()

    def TopicNetwork(self):
        with tf.name_scope("Topic_autoencoder") as scope:
            with tf.variable_scope("Encoder_1") as scope:
                cell_1_1 = tf.nn.rnn_cell.LSTMCell(num_units=256, reuse=False)
                h0 = cell_1_1.zero_state([config.batch_size], tf.float32)
                en1_A, state = tf.nn.dynamic_rnn(cell=cell_1_1, inputs=self.T_A, initial_state=h0)
                en1_A = tf.layers.batch_normalization(en1_A)
                cell_1_2 = tf.nn.rnn_cell.LSTMCell(num_units=256, reuse=True)
                h0 = cell_1_2.zero_state([config.batch_size], tf.float32)
                en1_B, state = tf.nn.dynamic_rnn(cell=cell_1_2, inputs=self.T_B, initial_state=h0)
                en1_B = tf.layers.batch_normalization(en1_B)
            with tf.variable_scope("Encoder_2") as scope:
                cell_2_1 = tf.nn.rnn_cell.LSTMCell(num_units=100, reuse=False)
                h0 = cell_2_1.zero_state([config.batch_size], tf.float32)
                en2_A, state = tf.nn.dynamic_rnn(cell=cell_2_1, inputs=en1_A, initial_state=h0)
                en2_A = tf.layers.batch_normalization(en2_A)
                cell_2_2 = tf.nn.rnn_cell.LSTMCell(num_units=100, reuse=True)
                h0 = cell_2_2.zero_state([config.batch_size], tf.float32)
                en2_B, state = tf.nn.dynamic_rnn(cell=cell_2_2, inputs=en1_B, initial_state=h0)
                en2_B = tf.layers.batch_normalization(en2_B)

            output_A_ = en2_A
            output_A = tf.transpose(output_A_, perm=[0, 2, 1])
            output_A = tf.reduce_mean(tf.matmul(output_A, output_A_), 2)
            output_B_ = en2_B
            output_B = tf.transpose(output_B_, perm=[0, 2, 1])
            output_B = tf.reduce_mean(tf.matmul(output_B, output_B_), 2)

            with tf.variable_scope("decoder_1") as scope:
                cell_3_1 = tf.nn.rnn_cell.LSTMCell(num_units=256, reuse=False)
                h0 = cell_3_1.zero_state([config.batch_size], tf.float32)
                de1_A, state = tf.nn.dynamic_rnn(cell=cell_3_1, inputs=en2_A, initial_state=h0)
                de1_A = tf.layers.batch_normalization(de1_A)
                cell_3_2 = tf.nn.rnn_cell.LSTMCell(num_units=256, reuse=True)
                h0 = cell_3_2.zero_state([config.batch_size], tf.float32)
                de1_B, state = tf.nn.dynamic_rnn(cell=cell_3_2, inputs=en2_B, initial_state=h0)
                de1_B = tf.layers.batch_normalization(de1_B)
            with tf.variable_scope("decoder_2") as scope:
                cell_4_1 = tf.nn.rnn_cell.LSTMCell(num_units=100, reuse=False)
                h0 = cell_4_1.zero_state([config.batch_size], tf.float32)
                de2_A, state = tf.nn.dynamic_rnn(cell=cell_4_1, inputs=de1_A, initial_state=h0)
                de2_A = tf.layers.batch_normalization(de2_A)
                cell_4_2 = tf.nn.rnn_cell.LSTMCell(num_units=100, reuse=True)
                h0 = cell_4_2.zero_state([config.batch_size], tf.float32)
                de2_B, state = tf.nn.dynamic_rnn(cell=cell_4_2, inputs=de1_B, initial_state=h0)
                de2_B = tf.layers.batch_normalization(de2_B)

            return output_A, output_B, de2_A, de2_B

    def compute_loss(self):
        p1 = tf.reduce_sum(self.gruA*self.gruB, 1)
        p1 = tf.log(tf.sigmoid(p1) + 0.001)

        p2 = tf.reduce_sum(self.N_A*self.N_B, 1)
        p2 = tf.log(tf.sigmoid(p2) + 0.001)

        p3 = tf.log(tf.sigmoid(tf.reduce_sum(self.N_A*self.gruB, 1)) + 0.001)

        p4 = tf.log(tf.sigmoid(tf.reduce_sum(self.N_B*self.gruA, 1)) + 0.001)
        # p5 = -tf.reduce_mean(self.T_A * tf.log(self.resA) + 0.001)
        #
        # p6 = -tf.reduce_mean(self.T_B * tf.log(self.resB) + 0.001)
        # tf.summary.scalar(name="Content Loss", tensor=p1)
        # tf.summary.scalar(name="Node Loss", tensor=p2)
        # tf.summary.scalar(name="Content B Node A Loss", tensor=p3)
        # tf.summary.scalar(name="Content A Node B Loss", tensor=p4)

        loss = -tf.reduce_sum(0.5 * p1 + 4 * p2 + 0.5 * p3 + 0.5*p4)
        return loss




if __name__ == '__main__':
    # c = model_keras(10, 1000)
    # m1 = c.Modelv2()
    # print(m1.summary())
    m = model_tf(vocab_size=1000, num_nodes=10)
    with tf.Session() as sess:
        # merge_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("./logs", sess.graph)
        # summary_str = sess.run(merge_op)
        # summary_writer.add_summary(summary_str)
