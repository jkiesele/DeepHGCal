from readers.InputReader import InputReader
import tensorflow as tf
from readers.DataAndNumEntriesReader import DataAndNumEntriesReader
import numpy as np

class DenseToConvReader(DataAndNumEntriesReader):
    def __init__(self, files_list, num_max_entries, num_data_dims, num_batch, repeat=True, shuffle_size=None):
        super(DenseToConvReader, self).__init__(files_list, num_max_entries, num_data_dims, num_batch, repeat, shuffle_size)
        self.construct_indexing_array()

    def _parse_function(self, example_proto):
        keys_to_features = {
            'data': tf.FixedLenFeature((self.num_max_entries + 1, self.num_data_dims), tf.float32),
        }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['data']

    def construct_indexing_array(self):
        self.indices_x = np.zeros(shape=(2679,1))
        self.indices_y = np.zeros(shape=(2679,1))
        self.indices_z = np.zeros(shape=(2679,1))
        self.indices_d = np.zeros(shape=(2679,1))

        a = np.arange(self.num_batch)
        b = np.tile(a[..., np.newaxis], reps=(1, 2679))[..., np.newaxis]
        m = tf.concat((self.indices_x, self.indices_y, self.indices_z, self.indices_d)[np.newaxis, ...])
        indexing_array = np.concatenate(b, np.tile(m, reps=[self.num_batch, 1, 1]), axis=2)
        self.indexing_array = indexing_array

    def get_feeds(self):
        """
        Returns the feeds (data, num_entries)

        :param files_list:
        :param num_batch:
        :param num_max_entries:
        :param num_data_dims:
        :param repeat:
        :param shuffle_size:
        :return:
        """
        with open(self.files_list) as f:
            content = f.readlines()
        file_paths = [x.strip() for x in content]
        dataset = tf.data.TFRecordDataset(file_paths, compression_type='GZIP')
        dataset = dataset.map(self._parse_function)
        dataset = dataset.shuffle(buffer_size=self.num_batch * 3 if self.shuffle_size is None else self.shuffle_size)
        dataset = dataset.repeat(None if self.repeat else 1)
        dataset = dataset.batch(self.num_batch)
        iterator = dataset.make_one_shot_iterator()
        data = iterator.get_next()
        data = data[:, 0:-1, :]
        num_entries = tf.ones(shape=(self.num_batch, 1) ,dtype=tf.int64) * self.num_max_entries

        return data, num_entries
