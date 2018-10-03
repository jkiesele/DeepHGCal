from readers.InputReader import InputReader
import tensorflow as tf
from readers.DataAndNumEntriesReader import DataAndNumEntriesReader

class FixNumEntriesReader(DataAndNumEntriesReader):
    def __init__(self, files_list, num_max_entries, num_data_dims, num_batch, repeat=True, shuffle_size=None):
        super(FixNumEntriesReader, self).__init__(files_list, num_max_entries, num_data_dims, num_batch, repeat, shuffle_size)

    def _parse_function(self, example_proto):
        keys_to_features = {
            'data': tf.FixedLenFeature((self.num_max_entries + 1, self.num_data_dims), tf.float32),
        }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['data']

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
        num_entries = tf.ones(shape=(self.num_batch) ,dtype=tf.int64) * self.num_max_entries

        return data, num_entries
