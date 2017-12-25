import tensorflow as tf

import config


def _input_fn(path, perform_shuffle=False, repeat_count=1):
    """
    Input Function for tf.Dataset Object
    :param path: source dataset dir
    :param perform_shuffle:
    :param repeat_count:
    :return:
    """

    def decode_txt(line):
        nodeA = []
        nodeB = []
        content = []
        time = []
        sentiment = []
        return (nodeA, nodeB, content, time, sentiment)

    ds = tf.data.TextLineDataset(path)
    ds = ds.map(decode_txt, num_parallel_calls=4)
    if perform_shuffle:
        ds = ds.shuffle(buffer_size=256)
    ds = ds.repeat(repeat_count)
    ds = ds.padded_batch(config.batch_size, [config.MAX_LEN])
    iterator = ds.make_one_shot_iterator()
    return iterator.get_next()


class model_tf(object):
    def __init__(self, vocab_size: int, num_nodes: int):
        with tf.name_scope('read_inputs') as scope:
            self.Text_a = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Text_a')
            self.Node_a = tf.placeholder(tf.int32, [config.batch_size], name='node_1')
            self.Node_b = tf.placeholder(tf.int32, [config.batch_size], name='node_2')
            self.Time_a = tf.placeholder(tf.int32, [config.batch_size, 1], name='time_1')
            self.Polarity_a = tf.placeholder(tf.int32, [config.batch_size, 2], name='polarity_a')

        with tf.name_scope('initialize_embeddings') as scope:
            self.text_embed = tf.Variable(tf.truncated_normal([vocab_size, 10], stddev=0.3))
            self.node_embed = tf.Variable(tf.truncated_normal([num_nodes, 16], stddev=0.3))
            self.time_embed = tf.Variable(tf.truncated_normal([1, 10], stddev=0.3))
            self.polarity_embed = tf.Variable(tf.truncated_normal([2, 5], stddev=0.3))
            self.node_embed = tf.clip_by_norm(self.node_embed, clip_norm=1, axes=1)

        with tf.name_scope('lookup_embeddings') as scope:
            self.Text = tf.nn.embedding_lookup(self.text_embed, self.Text_a)
            self.N_A = tf.nn.embedding_lookup(self.node_embed, self.Node_a)
            self.N_B = tf.nn.embedding_lookup(self.node_embed, self.Node_b)
            self.Time = tf.nn.embedding_lookup(self.time_embed, self.Time_a)
            self.Sentiment = tf.nn.embedding_lookup(self.polarity_embed, self.Polarity_a)
            # print(self.node_embed)


        # self.gruA, self.resA, = self.TopicNetwork()
        # self.loss = self.compute_loss()
        with tf.name_scope('tensor_process') as scope:
            self.Text = tf.layers.flatten(self.Text)
            self.Sentiment = tf.layers.flatten(self.Sentiment)
            self.Time = tf.layers.flatten(self.Time)
            # self.Sentiment = tf.squeeze(self.Sentiment, -1)
            # print(self.Text)
            # print(self.N_A)
            # print(self.N_B)
            # print(self.Sentiment)
            self.merge_tensor = tf.concat([self.Text, self.Time, self.Sentiment], axis=1)
            print(self.merge_tensor)

        # (self.contents_encoder_op, self.contents_decoder_op,
        #  self.time_encoder_op, self.time_decoder_op,
        #  self.polarity_encoder_op, self.polarity_decoder_op) = self.TopicAutoencoder()
        #self.loss = self.compute_loss_autoencoder()

        (self.encoder_op, self.decoder_op) = self.TopicAutoencoder()

        self.loss = self.compute_loss_merge()

    def TopicNetwork(self):
        with tf.name_scope("Topic_autoencoder") as scope:
            with tf.variable_scope("Encoder_1") as scope:
                cell_1_1 = tf.nn.rnn_cell.GRUCell(num_units=256, reuse=False)
                h0 = cell_1_1.zero_state([config.batch_size], tf.float32)
                en1_A, state = tf.nn.dynamic_rnn(cell=cell_1_1, inputs=self.T_A, initial_state=h0)
                en1_A = tf.layers.batch_normalization(en1_A)
                # cell_1_2 = tf.nn.rnn_cell.GRUCell(num_units=256, reuse=True)
                # h0 = cell_1_2.zero_state([config.batch_size], tf.float32)
                # en1_B, state = tf.nn.dynamic_rnn(cell=cell_1_2, inputs=self.T_B, initial_state=h0)
                # en1_B = tf.layers.batch_normalization(en1_B)
            with tf.variable_scope("Encoder_2") as scope:
                cell_2_1 = tf.nn.rnn_cell.GRUCell(num_units=100, reuse=False)
                h0 = cell_2_1.zero_state([config.batch_size], tf.float32)
                en2_A, state = tf.nn.dynamic_rnn(cell=cell_2_1, inputs=en1_A, initial_state=h0)
                en2_A = tf.layers.batch_normalization(en2_A)
                # cell_2_2 = tf.nn.rnn_cell.GRUCell(num_units=100, reuse=True)
                # h0 = cell_2_2.zero_state([config.batch_size], tf.float32)
                # en2_B, state = tf.nn.dynamic_rnn(cell=cell_2_2, inputs=en1_B, initial_state=h0)
                # en2_B = tf.layers.batch_normalization(en2_B)

            output_A_ = en2_A
            output_A = tf.transpose(output_A_, perm=[0, 2, 1])
            output_A = tf.reduce_mean(tf.matmul(output_A, output_A_), 2)
            # output_B_ = en2_B
            # output_B = tf.transpose(output_B_, perm=[0, 2, 1])
            # output_B = tf.reduce_mean(tf.matmul(output_B, output_B_), 2)

            with tf.variable_scope("decoder_1") as scope:
                cell_3_1 = tf.nn.rnn_cell.GRUCell(num_units=256, reuse=False)
                h0 = cell_3_1.zero_state([config.batch_size], tf.float32)
                de1_A, state = tf.nn.dynamic_rnn(cell=cell_3_1, inputs=en2_A, initial_state=h0)
                de1_A = tf.layers.batch_normalization(de1_A)
                # cell_3_2 = tf.nn.rnn_cell.GRUCell(num_units=256, reuse=True)
                # h0 = cell_3_2.zero_state([config.batch_size], tf.float32)
                # de1_B, state = tf.nn.dynamic_rnn(cell=cell_3_2, inputs=en2_B, initial_state=h0)
                # de1_B = tf.layers.batch_normalization(de1_B)
            with tf.variable_scope("decoder_2") as scope:
                cell_4_1 = tf.nn.rnn_cell.GRUCell(num_units=100, reuse=False)
                h0 = cell_4_1.zero_state([config.batch_size], tf.float32)
                de2_A, state = tf.nn.dynamic_rnn(cell=cell_4_1, inputs=de1_A, initial_state=h0)
                de2_A = tf.layers.batch_normalization(de2_A)
                # cell_4_2 = tf.nn.rnn_cell.GRUCell(num_units=100, reuse=True)
                # h0 = cell_4_2.zero_state([config.batch_size], tf.float32)
                # de2_B, state = tf.nn.dynamic_rnn(cell=cell_4_2, inputs=de1_B, initial_state=h0)
                # de2_B = tf.layers.batch_normalization(de2_B)

            return output_A, de2_A

    def TopicAutoencoder(self) -> (tf.Tensor, tf.Tensor):
        n_input = 220
        n_hidden_1 = 128
        n_hidden_2 = 64
        n_hidden_3 = 16
        with tf.name_scope("init_weights") as scope:
            weights = {
                'encoder_w1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.3)),
                'encoder_w2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.3)),
                'encoder_w3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.3)),
                'decoder_w1': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2], stddev=0.3)),
                'decoder_w2': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], stddev=0.3)),
                'decoder_w3': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.3))
            }
            biases = {
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.3)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.3)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.3)),
                'decoder_b1': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.3)),
                'decoder_b2': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.3)),
                'decoder_b3': tf.Variable(tf.truncated_normal([n_input], stddev=0.3))
            }

        def encoder(x: tf.Tensor) -> tf.Tensor:
            with tf.name_scope("encoder") as scope:
                layer_1 = tf.nn.relu6(tf.nn.xw_plus_b(x, weights['encoder_w1'], biases['encoder_b1']))
                layer_2 = tf.nn.relu6(tf.nn.xw_plus_b(layer_1, weights['encoder_w2'], biases['encoder_b2']))
                layer_3 = tf.nn.relu6(tf.nn.xw_plus_b(layer_2, weights['encoder_w3'], biases['encoder_b3']))
                return layer_3

        def decoder(x: tf.Tensor) -> tf.Tensor:
            with tf.name_scope("decoder") as scope:
                layer_1 = tf.nn.relu6(tf.nn.xw_plus_b(x, weights['decoder_w1'], biases['decoder_b1']))
                layer_2 = tf.nn.relu6(tf.nn.xw_plus_b(layer_1, weights['decoder_w2'], biases['decoder_b2']))
                layer_3 = tf.nn.relu6(tf.nn.xw_plus_b(layer_2, weights['decoder_w3'], biases['decoder_b3']))
                return layer_3
        #
        # # Contents
        # contents_encoder_op = encoder(self.Text)
        # contents_decoder_op = decoder(contents_encoder_op)
        #
        # # Time
        # time_encoder_op = encoder(self.Time)
        # time_decoder_op = decoder(time_encoder_op)
        #
        # # Polarity
        # polarity_encoder_op = encoder(self.Sentiment)
        # polarity_decoder_op = decoder(polarity_encoder_op)

        # return (contents_encoder_op, contents_decoder_op,
        #         time_encoder_op, time_decoder_op,
        #         polarity_encoder_op, polarity_decoder_op)

        encoder_op = encoder(self.merge_tensor)
        decoder_op = decoder(encoder_op)

        return encoder_op, decoder_op

    def compute_loss(self):
        p1 = tf.reduce_sum(self.gruA * self.gruB, 1)
        p1 = tf.log(tf.sigmoid(p1) + 0.001)

        p2 = tf.reduce_sum(self.N_A * self.N_B, 1)
        p2 = tf.log(tf.sigmoid(p2) + 0.001)

        p3 = tf.log(tf.sigmoid(tf.reduce_sum(self.N_A * self.gruB, 1)) + 0.001)

        p4 = tf.log(tf.sigmoid(tf.reduce_sum(self.N_B * self.gruA, 1)) + 0.001)
        # p5 = -tf.reduce_mean(self.T_A * tf.log(self.resA) + 0.001)
        #
        # p6 = -tf.reduce_mean(self.T_B * tf.log(self.resB) + 0.001)
        # tf.summary.scalar(name="Content Loss", tensor=p1)
        # tf.summary.scalar(name="Node Loss", tensor=p2)
        # tf.summary.scalar(name="Content B Node A Loss", tensor=p3)
        # tf.summary.scalar(name="Content A Node B Loss", tensor=p4)

        loss = -tf.reduce_sum(0.5 * p1 + 4 * p2 + 0.5 * p3 + 0.5 * p4)
        return loss

    def compute_loss_autoencoder(self) -> tf.Tensor:
        # Contents Autoencoder Reconstruction Loss
        loss_contents_rec = tf.reduce_mean(tf.pow(self.Text - self.contents_decoder_op, 2))

        # Time Autoencoder Reconstruction Loss
        loss_time_rec = tf.reduce_mean(tf.pow(self.Time - self.time_decoder_op, 2))

        # Sentiment Autoencoder Reconstruction Loss
        loss_sentiment_rec = tf.reduce_mean(tf.pow(self.Sentiment - self.polarity_decoder_op, 2))

        # All over Reconstruction Loss
        loss_all_rec = tf.reduce_sum(loss_contents_rec + loss_sentiment_rec + loss_time_rec)

        node_loss = tf.log(tf.sigmoid(tf.reduce_sum(self.N_A * self.N_B, 1)) + 0.001)

        contents_loss = tf.log(tf.sigmoid(tf.reduce_sum(self.N_B * self.contents_encoder_op, 1)) + 0.001)

        time_loss = tf.log(tf.sigmoid(tf.reduce_sum(self.N_B * self.time_encoder_op, 1)) + 0.001)

        loss = -tf.reduce_sum(node_loss + contents_loss + time_loss) + loss_all_rec

        return loss

    def compute_loss_merge(self) -> tf.Tensor:
        loss_rec = tf.reduce_mean(tf.pow(self.merge_tensor - self.decoder_op, 2))

        node_loss = tf.log(tf.sigmoid(tf.reduce_sum(self.N_A * self.N_B)) + 0.001)

        embedding_loss = tf.log(tf.sigmoid(tf.reduce_sum(self.N_B * self.encoder_op)) + 0.001)

        loss = -tf.reduce_sum(node_loss + embedding_loss) + loss_rec

        return loss

if __name__ == '__main__':
    c = model_tf(20, 2412)
