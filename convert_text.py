import os
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
# from smart_open import smart_open
import gensim

model = gensim.models.KeyedVectors.\
    load_word2vec_format(fname=os.path.join("C:/data", "GoogleNews-vectors-negative300.bin"), binary=True)

print("Load Word2Vec model Success!")

user_file_path = './data/twitter_network'
src_data_path = './'
dst_data_path = './'
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

    for data in openfile(os.path.join(user_file_path, "user_map.txt")):
        name = data.split(' ')[1]
        id = data.split(' ')[0]
        user_map[name] = id
    print("Preload user map success!")

    with open(os.path.join(user_file_path, "user_list.txt")) as f:
        count = 0
        while 1:
            data = f.readline().replace("\n", "")
            if not data:
                break
            user_list[data] = count
            count += 1
        print("Load user list success!")

    temp_ = []
    temp__1 = user_map.values()
    temp__2 = user_list.keys()
    for i_ in temp__1:
        if i_ not in temp__2:
            temp_.append(i_)

    user_map.clear()
    c = 0
    sum = []
    with open(os.path.join(user_file_path, "user_map.txt")) as f:
        while 1:
            data = f.readline().replace("\n", "")
            if not data:
                break
            name = data.split(' ')[1]
            id = data.split(' ')[0]
            if id in temp_:
                sum.append(c)
                c += 1
                continue
            user_map[name] = id
            c += 1
        print("Load user map success!")
    return sum


def process(src_path, dst_path, ignore_list):
    with open(os.path.join(src_path, "name.txt")) as src:
        with open(os.path.join(dst_path, "name.txt"), 'w') as dst:
            count = 0
            while 1:
                if count in ignore_list:
                    count += 1
                    continue
                name = src.readline().replace("\n", '')
                if name == '':
                    break
                id_ = user_list[user_map[name]]
                dst.write(id_ + '\n')
                count += 1

    with open(os.path.join(src_path, "time.txt")) as src:
        with open(os.path.join(dst_path, "time.txt"), 'w') as dst:
            count = 0
            while 1:
                if count in ignore_list:
                    count += 1
                    continue
                data = src.readline()
                if data.replace('\n', '') == '':
                    break
                dst.write(data)

    with open(os.path.join(src_path, "tweet_id.txt")) as src:
        with open(os.path.join(dst_path, "tweet_id.txt"), 'w') as dst:
            count = 0
            while 1:
                if count in ignore_list:
                    count += 1
                    continue
                data = src.readline()
                if data.replace('\n', '') == '':
                    break
                dst.write(data)

    with open(os.path.join(src_path, "content.txt")) as src:
        with open(os.path.join(dst_path, "polarity.txt"), 'w') as dst_1:
            with open(os.path.join(dst_path, "content.txt"), 'w') as dst_2:
                count = 0
                while 1:
                    if count in ignore_list:
                        count += 1
                        continue
                    data = src.readline()
                    if data.replace('\n', '') == '':
                        break
                    temp_data = []
                    for i in data.split(" "):
                        temp_data.append(word_map[i])
                    data = " ".join(temp_data)
                    blob = TextBlob(data, analyzer=NaiveBayesAnalyzer())
                    dst_1.write("{} {}\n".format(blob.sentiment.p_pos, blob.sentiment.p_neg))
                    temp_vector = []
                    for i in data:
                        temp_vector.append(model.wv[i])
                    data = " ".join(temp_vector)
                    dst_2.write("{}\n".format(data))

    with open(os.path.join(src_path, "reply_to_user_file.txt")) as src:
        with open(os.path.join(dst_path, "reply_to_user_file.txt"), 'w') as dst:
            count = 0
            while 1:
                if count in ignore_list:
                    count += 1
                    continue
                name = src.readline().replace("\n", '')
                if name == '':
                    break
                id_ = user_list[user_map[name]]
                dst.write(id_ + '\n')
                count += 1

    with open(os.path.join(src_path, "retweet.txt")) as src:
        with open(os.path.join(dst_path, "retweet.txt"), 'w') as dst:
            count = 0
            while 1:
                if count in ignore_list:
                    count += 1
                    continue
                name = src.readline().replace("\n", '')
                if name == '':
                    break
                id_ = user_list[user_map[name]]
                dst.write(id_ + '\n')
                count += 1


if __name__ == '__main__':
    ignore_list = load_map()
    # process(src_data_path, dst_data_path, ignore_list)
