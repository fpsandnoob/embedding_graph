import config
import os
import numpy as np
from tensorflow.contrib import learn
from negativeSample import InitNegTable
import random

class dataset:
    def __init__(self, dir_path):
        self.node = os.path.join(dir_path, "vector.txt")
        self.time = os.path.join(dir_path, "time_.txt")
        self.polarity = os.path.join(dir_path, "polarity.txt")
        self.contents = os.path.join(dir_path, "contents.txt")

        self.edges = self.load_edges

    def load_edges(self):
        with open(self.node, encoding='utf-8') as f:
            while 1:
                graph_file = f.readline()
                if graph_file == "\n":
                    break
                edges = list(map(int, graph_file.strip().replace("\n", "").split('\t')))
                yield edges

    def load_contents(self):
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
        with open(self.polarity, encoding='utf-8') as f:
            while 1:
                data = f.readline()
                temp_contents = data.replace('\n', '')
                if temp_contents == "":
                    break
                contents = list(map(float, temp_contents.split(" ")))
                yield contents

    def load_time(self):
        with open(self.time, encoding='utf-8') as f:
            while 1:
                data = f.readline()
                temp_contents = data.replace('\n', '')
                if temp_contents == "":
                    break
                contents = list(map(float, temp_contents.split(" ")))
                yield contents

if __name__ == '__main__':
    path = r'D:\data\new_data'
    c = dataset(path)
    for x in c.load_time():
        print(x)
