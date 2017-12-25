import os
from textblob.sentiments import NaiveBayesAnalyzer
import numpy as np
import csv
from datetime import datetime
# from smart_open import smart_open
# import numpy as np
# import gensim
naiveBayesAnalyzer = NaiveBayesAnalyzer()
# model = gensim.models.KeyedVectors.\
#     load_word2vec_format(fname=os.path.join("C:/data", "GoogleNews-vectors-negative300.bin"), binary=True)
#
# print("Load Word2Vec model Success!")

user_file_path = './data/twitter_network'
src_data_path = 'D:/data/embedding_graph/'
dst_data_path = 'D:/data/new_data/'
word_table_path = './data/WordTable.txt'

user_map = {}
user_list = {}
word_map = {}


def openfile(path):
    with open(path) as f:
        while 1:
            data = f.readline().replace('\n', '')
            if not data:
                break
            yield data


def openfilec(path):
    with open(path) as f:
        count = 0
        while 1:
            data = f.readline().replace('\n', '')
            if not data:
                break
            yield data


def load_map():
    for data in openfile(word_table_path):
        if len(data.split('\t')) == 1:
            continue
        id = data.split('\t')[0]
        word = data.split('\t')[2]
        word_map[id] = word
    print("Load word map success!")

    with open(os.path.join(src_data_path, "name.txt"), encoding='utf-8') as src_1:
        with open(os.path.join(src_data_path, "reply_to_user_file.txt"), encoding='utf-8') as src_2:
            with open(os.path.join(src_data_path, "retweet.txt"), encoding='utf-8') as src_3:
                count = 0
                while 1:
                    name = src_1.readline()
                    if name == '':
                        break
                    if name.replace('\n', '') == '':
                        continue
                    if name.replace('\n', '') not in user_map.keys():
                        user_map[name.replace('\n', '')] = str(count)
                        count += 1
                    else:
                        continue
                while 1:
                    name = src_2.readline()
                    if name == '':
                        break
                    if name.replace('\n', '') == '':
                        continue
                    if name.replace('\n', '') not in user_map.keys():
                        user_map[name.replace('\n', '')] = str(count)
                        count += 1
                    else:
                        continue
                while 1:
                    name = src_3.readline()
                    if name == '':
                        break
                    if name.replace('\n', '') == '':
                        continue
                    if name.replace('\n', '') not in user_map.keys():
                        user_map[name.replace('\n', '')] = str(count)
                        count += 1
                    else:
                        continue


    # for data in openfile(os.path.join(user_file_path, "user_map.txt")):
    #     name = data.split(' ')[1]
    #     id = data.split(' ')[0]
    #     user_map[name] = id
    # print("Preload user map success!")
    #
    # with open(os.path.join(user_file_path, "user_list.txt")) as f:
    #     count = 0
    #     while 1:
    #         data = f.readline().replace("\n", "")
    #         if not data:
    #             break
    #         user_list[data] = count
    #         count += 1
    #     print("Load user list success!")
    #
    # temp_ = []
    # temp__1 = user_map.values()
    # temp__2 = user_list.keys()
    # for i_ in temp__1:
    #     if i_ not in temp__2:
    #         temp_.append(i_)
    #
    # user_map.clear()
    # c = 0
    # sum = []
    # with open(os.path.join(user_file_path, "user_map.txt")) as f:
    #     while 1:
    #         data = f.readline().replace("\n", "")
    #         if not data:
    #             break
    #         name = data.split(' ')[1]
    #         id = data.split(' ')[0]
    #         if id in temp_:
    #             sum.append(c)
    #             c += 1
    #             continue
    #         user_map[name] = id
    #         c += 1
    #     print("Load user map success!")
    # return sum

def process(src_path, dst_path):
    # with open(os.path.join(src_path, "name.txt"), encoding='utf-8') as src:
    #     with open(os.path.join(dst_path, "name.txt"), 'w', encoding='utf-8') as dst:
    #         count = 0
    #         while 1:
    #             count += 1
    #             name = src.readline()
    #             if name == '':
    #                 break
    #             if name.replace('\n', '') == '':
    #                 continue
    #             dst.write(user_map[name.replace('\n', '')] + '\n')
    #             count += 1
    #
    # with open(os.path.join(src_path, "time.txt"), encoding='utf-8') as src:
    #     with open(os.path.join(dst_path, "time.txt"), 'w', encoding='utf-8') as dst:
    #         count = 0
    #         while 1:
    #             data = src.readline()
    #             if data.replace('\n', '') == '':
    #                 break
    #             dst.write(data)
    #
    # with open(os.path.join(src_path, "tweet_id.txt"), encoding='utf-8') as src:
    #     with open(os.path.join(dst_path, "tweet_id.txt"), 'w', encoding='utf-8') as dst:
    #         count = 0
    #         while 1:
    #             data = src.readline()
    #             if data.replace('\n', '') == '':
    #                 break
    #             dst.write(data)

    # with open(os.path.join(src_path, "contents.txt")) as src:
    #     with open(os.path.join(dst_path, "polarity.txt"), 'w') as dst_1:
    #         with open(os.path.join(dst_path, "polarity.txt"), 'w') as dst_2:
    #             count = 0
    #             # data_ = []
    #             while 1:
    #                 data = src.readline()
    #                 if data == '':
    #                     break
    #                 temp_data = []
    #                 __ = data.replace('\n', '').rstrip()
    #                 for i in __.split(" "):
    #                     if i == '':
    #                         continue
    #                     temp_data.append(word_map[i])
    #                 data = " ".join(temp_data)
    #                 blob = TextBlob(data, analyzer=naiveBayesAnalyzer)
    #                 dst_1.write("{} {}\n".format(blob.sentiment.p_pos, blob.sentiment.p_neg))
    #                 print(count)
    #                 count += 1
                    # temp_vector = np.array([0], dtype=np.float64)
                    # for i in temp_data:
                    #     try:
                    #         np.concatenate((temp_vector, model.wv[i]))
                    #     except KeyError:
                    #         np.concatenate((temp_vector, np.zeros([300])))
                    # print(np.shape(temp_vector[1:]))
                    # np.savetxt(dst_2, temp_vector[1:])
                # data_.append(data)
            # data_ = np.array(data_)
            # np.save(os.path.join(dst_path, "contents.npy"), data_)

    # with open(os.path.join(src_path, "reply_to_user_file.txt"), encoding='utf-8') as src:
    #     with open(os.path.join(dst_path, "reply_to_user_file.txt"), 'w', encoding='utf-8') as dst:
    #         count = 0
    #         while 1:
    #             name = src.readline()
    #             if name == '':
    #                 break
    #             if name.replace('\n', '') == '':
    #                 continue
    #             id_ = user_map[name.replace('\n', '')]
    #             dst.write(id_ + '\n')
    #             count += 1
    #
    # with open(os.path.join(src_path, "retweet.txt"), encoding='utf-8') as src:
    #     with open(os.path.join(dst_path, "retweet.txt"), 'w', encoding='utf-8') as dst:
    #         count = 0
    #         while 1:
    #             name = src.readline()
    #             if name == '':
    #                 break
    #             # if name.replace('\n', '') == '':
    #             #     continue
    #             id_ = user_map[name.replace('\n', '')]
    #             dst.write(id_ + '\n')
    #             count += 1

        with open(os.path.join(src_path, "retweet.txt"), encoding='utf-8') as src_1:
            with open(os.path.join(src_path, "reply_to_user_file.txt"), encoding='utf-8') as src_2:
                with open(os.path.join(dst_path, "node.txt"), encoding='utf-8', mode='w') as dst:
                    while 1:
                        name_1 = src_1.readline()
                        name_2 = src_2.readline()
                        if name_1 == '':
                            break
                        if name_1.replace('\n', '') != '-1':
                            dst.write(user_map[name_1.replace('\n', '')] + '\n')
                            continue
                        elif name_2.replace('\n', '') != '-1':
                            dst.write(user_map[name_2.replace('\n', '')] + '\n')
                            continue
                        else:
                            raise ValueError


def generate_Edge(dst_path):
    count = 0
    with open(os.path.join(dst_path, "node.txt"), encoding='utf-8') as node_B:
        with open(os.path.join(dst_path, "name.txt"), encoding='utf-8') as node_A:
            with open(os.path.join(dst_path, 'vector.txt'), mode='w') as dst:
                while 1:
                    print(count)
                    node_a = node_A.readline()
                    node_b = node_B.readline()
                    if node_a.replace('\n', '') == '':
                        break
                    dst.write("{}\t{}\n".format(node_a.replace('\n', ''), node_b.replace('\n', '')))
                    count += 1

def parser_data(dst_path):
    with open(os.path.join(dst_path, "time.txt"), encoding='utf-8') as src:
        with open(os.path.join(dst_path, "time_.txt"), "w", encoding='utf-8') as dst:
            while 1:
                src_str = src.readline().replace('\n', '')
                if src_str == '':
                    break
                try:
                    dt = datetime.strptime(src_str, "%a %b %d %X %z %Y")
                except ValueError:
                    dt = datetime.strptime(src_str, "%a %b %d %X %Z %Y")
                dst.write(str(dt.timestamp()) + '\n')


if __name__ == '__main__':
    # load_map()
    # process(src_data_path, dst_data_path)
    # generate_Edge(dst_data_path)
    parser_data(dst_data_path)
