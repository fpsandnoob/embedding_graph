import config
import os
import numpy as np
from tensorflow.contrib import learn
from negativeSample import InitNegTable
import random


class dataset:
    def __init__(self, dir_path):
        self.num_nodes = 2658058
        self.num_vocab = 8177600

        self.nodes = os.path.join(dir_path, "vector.txt")
        self.times = os.path.join(dir_path, "time_.txt")
        self.polarities = os.path.join(dir_path, "polarity.txt")
        self.contents = os.path.join(dir_path, "contents.txt")

        self.edges = self.load_edges()
        self.content = self.load_contents()
        self.polarity = self.load_polarity()
        self.time = self.load_time()

    def load_edges(self):
        while 1:
            with open(self.nodes, encoding='utf-8') as f:
                while 1:
                    graph_file = f.readline()
                    if graph_file == "":
                        break
                    edges = list(map(int, graph_file.strip().replace("\n", "").split('\t')))
                    yield edges

    def load_contents(self):
        while 1:
            with open(self.contents, encoding='utf-8') as f:
                count = 1
                while 1:
                    if count == 39787305:
                        break
                    data = f.readline()
                    temp_contents = data.replace('\n', '')
                    if temp_contents == '':
                        contents = ''
                    else:
                        contents = list(map(int, temp_contents.strip().split(" ")))
                    word_ids = np.zeros(config.MAX_LEN, np.int64)
                    for idx, tokens in enumerate(contents):
                        if idx >= config.MAX_LEN:
                            break
                        word_ids[idx] = tokens
                    count += 1
                    yield word_ids

    def load_polarity(self):
        while 1:
            with open(self.polarities, encoding='utf-8') as f:
                while 1:
                    data = f.readline()
                    temp_contents = data
                    if temp_contents == "":
                        break
                    contents = list(map(float, temp_contents.strip().replace("\n", "").split(' ')))
                    yield contents

    def load_time(self):
        while 1:
            with open(self.times, encoding='utf-8') as f:
                while 1:
                    data = f.readline()
                    temp_contents = data
                    if temp_contents == "":
                        break
                    contents = list(map(float, temp_contents.strip().replace("\n", "").split(' ')))
                    yield contents

    # def negative_sample(self, edges):
    #     node1, node2 = zip(*edges)
    #     sample_edges = []
    #     func = lambda: self.negative_table[random.randint(0, config.neg_table_size - 1)]
    #     for i in range(len(edges)):
    #         neg_node = func()
    #         while node1[i] == neg_node or node2[i] == neg_node:
    #             neg_node = func()
    #         sample_edges.append([node1[i], node2[i], neg_node])
    #
    #     return sample_edges

    # def generate_batches(self, mode=None):
    #     edges = list(self.edges)
    #     num_batch = len(edges) / config.batch_size
    #     if mode == 'add':
    #         num_batch += 1
    #         edges.extend(edges[:(config.batch_size - len(edges) // config.batch_size)])
    #     if mode != 'add':
    #         random.shuffle(edges)
    #     sample_edges = edges[:int(num_batch * config.batch_size)]
    #     # sample_edges = self.negative_sample(sample_edges)
    #
    #     batches = []
    #     for i in range(int(num_batch)):
    #         batches.append(sample_edges[i * config.batch_size:(i + 1) * config.batch_size])
    #     # print sample_edges[0]
    #     return batches

    def generate_batches(self, mode=None):
        while 1:
            node_a = []
            node_b = []
            contents = []
            time = []
            polarity = []
            for i in range(config.batch_size):
                node_a.append(next(self.edges)[0])
                node_b.append(next(self.edges)[1])
                contents.append(next(self.content))
                time.append(next(self.time))
                polarity.append(next(self.polarity))
            yield np.array(node_a), np.array(node_b), np.array(contents), np.array(time, dtype=np.float64), np.array(polarity)

if __name__ == '__main__':
    path = r'D:\data\new_data'
    c = dataset(path)
    try:
        for c, i in enumerate(c.load_edges()):
            pass
    except ValueError:
        print(c)