import tensorflow as tf
import config

def input_fn(path, perform_shuffle=False, repeat_count=1):
    """
    Input Function for tf.Dataset Object
    :param path: source dataset dir
    :param perform_shuffle:
    :param repeat_count:
    :return:
    """

    def decode_txt(line):
        data = []
        return data

    ds = tf.data.TextLineDataset(path)
    ds = ds.map(decode_txt, num_parallel_calls=4)
    if perform_shuffle:
        ds = ds.shuffle(buffer_size=256)
    ds = ds.repeat(repeat_count)
    ds = ds.padded_batch(config.batch_size, [config.MAX_LEN])
    iterator = ds.make_one_shot_iterator()
    return iterator.get_next()
