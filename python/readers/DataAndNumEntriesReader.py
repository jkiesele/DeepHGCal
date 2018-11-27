from readers.InputReader import InputReader
import tensorflow as tf


class DataAndNumEntriesReader:
    def __init__(self, files_list, num_max_entries, num_data_dims, num_batch, repeat=True, shuffle_size=None):
        self.files_list = files_list
        self.repeat = repeat
        self.shuffle_size = shuffle_size
        self.num_max_entries = num_max_entries
        self.num_data_dims = num_data_dims
        self.num_batch = num_batch

    def get_feeds(self, shuffle=True):
        def _parse_function(example_proto):
            keys_to_features = {
                'data': tf.FixedLenFeature((self.num_max_entries, self.num_data_dims), tf.float32),
                'num_entries': tf.FixedLenFeature(1, tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, keys_to_features)
            return parsed_features['data'], parsed_features['num_entries']

        with open(self.files_list) as f:
            content = f.readlines()
        file_paths = [x.strip() for x in content]
        dataset = tf.data.TFRecordDataset(file_paths, compression_type='GZIP')
        dataset = dataset.map(_parse_function)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.num_batch * 3 if self.shuffle_size is None else self.shuffle_size)
        dataset = dataset.repeat(None if self.repeat else 1)
        dataset = dataset.batch(self.num_batch)
        iterator = dataset.make_one_shot_iterator()
        inputs = iterator.get_next()

        return inputs
