import numpy as np
import tensorflow as tf
from progressbar import ProgressBar
import train_dataset
import config
from embedding_graph_dataset import dataset

# load data
path = r'D:\data\new_data'
p = ProgressBar(max_value=9713)
data = dataset(path)

# start session

with tf.Graph().as_default():
    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth = True
    sess = tf.Session(config=config_)
    with sess.as_default():
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            model = train_dataset.model_tf(data.num_vocab, data.num_nodes)
            global_step = tf.Variable(0, trainable=False)
            # c = tf.train.exponential_decay(config.lr, global_step, 30, 0.9, staircase=False)
            opt = tf.train.AdamOptimizer(learning_rate=0.01)
            train_op = opt.minimize(model.loss, global_step)
            sess.run(tf.global_variables_initializer())
            # summary_writer = tf.summary.FileWriter("./logs", )

            # training
            print('start training.......')

            for epoch in range(config.num_epoch):
                loss_epoch = 0
                p.init()
                for _c in p(range(9713)):
                    node_a, node_b, contents, time, polarity = next(data.generate_batches())

                    feed_dict = {
                        model.Text_a: contents,
                        model.Node_a: node_a,
                        model.Node_b: node_b,
                        model.Time_a: time,
                        model.Polarity_a: polarity
                    }

                    # run the graph
                    # merge = tf.summary.merge_all()
                    _, loss_batch = sess.run([train_op, model.loss], feed_dict=feed_dict)

                    loss_epoch += loss_batch

                    # def variable_summaries(var, name):
                    #     with tf.name_scope('summaries'):
                    #         mean = tf.reduce_mean(var)
                    #         tf.summary.scalar('mean/' + name, mean)
                    # variable_summaries(loss_epoch, "loss")
                print('epoch: ', epoch + 1, ' loss: ', loss_epoch)

        # file = open('embed.txt', 'wb')
        # batches = data.generate_batches(mode='add')
        # num_batch = len(batches)
        # embed = [[] for _ in range(data.num_nodes)]
        # for i in range(num_batch):
        #     batch = batches[i]
        #
        #     node1, node2, node3 = zip(*batch)
        #     node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
        #     text1, text2, text3 = data.text[node1], data.text[node2], data.text[node3]
        #
        #     feed_dict = {
        #         model.Text_a: text1,
        #         model.Text_b: text2,
        #         # model.Text_neg: text3,
        #         model.Node_a: node1,
        #         model.Node_b: node2,
        #         # model.Node_neg: node3
        #     }

            # run the graph
            # convA, convB, TA, TB = sess.run([model.gruA, model.gruB, model.N_A, model.N_B], feed_dict=feed_dict)
        #     for i in range(config.batch_size):
        #         em = list(convA[i]) + list(TA[i])
        #         embed[node1[i]].append(em)
        #         em = list(convB[i]) + list(TB[i])
        #         embed[node2[i]].append(em)
        # for i in range(data.num_nodes):
        #     if embed[i]:
        #         # print embed[i]
        #         tmp = np.sum(embed[i], axis=0) / len(embed[i])
        #         str_ = ' '.join(map(str, tmp)) + '\n'
        #         file.write(bytes(str_, encoding='utf-8'))
        #     else:
        #         file.write(bytes("\n", encoding='utf-8'))
