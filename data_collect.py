import os

data_path = './data/'
save_path = 'D:/data/'


name = []
tweet_id = []
time = []
retweet = []
reply_to_user = []
reply_to_tweet_id = []
contents = []


def _write_to_file(dir_path, data):
    name_file = os.path.join(dir_path, "name.txt")
    tweet_id_file = os.path.join(dir_path, "tweet_id.txt")
    time_file = os.path.join(dir_path, "time.txt")
    retweet_file = os.path.join(dir_path, "retweet.txt")
    reply_to_user_file = os.path.join(dir_path, "reply_to_user_file.txt")
    contents_file = os.path.join(dir_path, "contents.txt")
    with open(name_file, mode="a", encoding='utf-8') as f:
        f.writelines(data[0])
    with open(tweet_id_file, mode="a", encoding='utf-8') as f:
        f.writelines(data[1])
    with open(time_file, mode="a", encoding='utf-8') as f:
        f.writelines(data[2])
    with open(retweet_file, mode="a", encoding='utf-8') as f:
        f.writelines(data[3])
    with open(reply_to_user_file, mode="a", encoding='utf-8') as f:
        f.writelines(data[4].split("\t")[0] + "\n")
    with open(contents_file, mode="a", encoding='utf-8') as f:
        f.writelines(data[5])


def collect(path):
    datas = parser_data(path)
    c = 0
    for data in datas:
        c += 1
        print(c)
        # name.append(data[0])
        # tweet_id.append(data[1])
        # time.append(data[2])
        # retweet.append(data[3])
        # reply_to_user.append(data[4].split("\t")[0] + "\n")
        # contents.append(data[5])
        _write_to_file(save_path, data)



def parser_data(path):
    with open(path, encoding='utf-8') as f:
        while 1:
            temp_name = f.readline()
            if temp_name == "\n":
                print('!!!!!!!')
                break
            temp_tweet_id = f.readline()
            temp_time = f.readline()
            f.readline()
            temp_retweet = f.readline()
            temp_reply = f.readline()
            temp_content = f.readline()
            count = f.readline().replace("\n", '')
            for _ in range(int(count) + 1):
                f.readline()
            yield [temp_name, temp_tweet_id, temp_time, temp_retweet, temp_reply, temp_content]


if __name__ == '__main__':
    for dirpath, dirnames, filenames in os.walk(data_path):
        # collect(data_path+"tweet_result_0_.txt")
        for f in filenames:
            if len(f) >= 18:
                path = os.path.join(dirpath, f)
                print(path)
                collect(path)

