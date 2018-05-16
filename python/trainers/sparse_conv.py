import tensorflow as tf
from models.sparse_conv import SparseConv
import numpy as np
import os
import configparser as cp


class SparseConvTrainer:
    def read_config(self, config_file_path, config_name):
        config_file = cp.ConfigParser()
        config_file.read(config_file_path)
        self.config = config_file[config_name]

    def __init__(self, config_file, config_name):
        self.read_config(config_file, config_name)

        self.from_scratch = int(self.config['from_scratch'])==1
        self.model_path = self.config['model_path']
        self.summary_path = self.config['summary_path']
        self.train_for_iterations = int(self.config['train_for_iterations'])
        self.save_after_iterations = int(self.config['save_after_iterations'])

        self.num_batch = int(self.config['batch_size'])
        self.num_spatial_features = int(self.config['input_spatial_features'])
        self.num_all_features = int(self.config['input_all_features'])
        self.num_max_entries = int(self.config['max_entries'])
        self.num_classes = int(self.config['num_classes'])
        self.num_max_neighbors = int(self.config['max_neighbors'])
        self.learning_rate = float(self.config['learning_rate'])
        self.training_files = self.config['training_files_list']
        self.validation_files = self.config['validation_files_list']
        self.validate_after = int(self.config['validate_after'])

        self.model = SparseConv(
            self.num_spatial_features,
            self.num_all_features,
            self.num_max_neighbors,
            self.num_batch,
            self.num_max_entries,
            self.num_classes,
            self.learning_rate
        )

        self.model.initialize()
        self.saver_all = tf.train.Saver() # TODO: Might want to move variables etc to a scope or something?



    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def clean_summary_dir(self):
        print("Cleaning summary dir")
        for the_file in os.listdir(self.summary_path):
            file_path = os.path.join(self.summary_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def test(self):
        model = SparseConv(3,10,5,10,1000)
        model.initialize()

    def __get_input_feeds(self, files_list):
        def _parse_function(example_proto):
            keys_to_features = {
                'space_features': tf.FixedLenFeature((self.num_max_entries, self.num_spatial_features), tf.float32),
                'all_features': tf.FixedLenFeature((self.num_max_entries, self.num_all_features), tf.float32),
                'neighbor_matrix': tf.FixedLenFeature((self.num_max_entries, self.num_max_neighbors), tf.int64),
                'labels_one_hot': tf.FixedLenFeature((self.num_classes), tf.int64),
                'num_entries': tf.FixedLenFeature(1, tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, keys_to_features)
            return parsed_features['space_features'], parsed_features['all_features'], parsed_features[
                'neighbor_matrix'], parsed_features['labels_one_hot'], parsed_features['num_entries']

        with open(files_list) as f:
            content = f.readlines()
        file_paths = [x.strip() for x in content]
        dataset = tf.data.TFRecordDataset(file_paths, compression_type='GZIP')
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.num_batch)
        iterator = dataset.make_one_shot_iterator()
        inputs = iterator.get_next()

        return inputs

    def train(self):
        placeholders = self. model.get_placeholders()
        graph_loss = self.model.get_losses()
        graph_optmiser = self.model.get_optimizer()
        graph_summary = self.model.get_summary()
        graph_summary_validation = self.model.get_summary_validation()
        graph_accuracy = self.model.get_accuracy()
        graph_logits, graph_prediction = self.model.get_compute_graphs()

        if self.from_scratch:
            self.clean_summary_dir()

        inputs_feed = self.__get_input_feeds(self.training_files)
        inputs_validation_feed = self.__get_input_feeds(self.validation_files)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        with tf.Session() as sess:
            sess.run(init)

            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

            if not self.from_scratch:
                self.saver_all.restore(sess, self.model_path)
                print("\n\nINFO: Loading model\n\n")
                with open(self.model_path + '.txt', 'r') as f:
                    iteration_number = int(f.read())
            else:
                iteration_number = 0

            print("Starting iterations")
            while iteration_number < self.train_for_iterations:
                inputs_train = sess.run(list(inputs_feed))

                inputs_train_dict = {
                    placeholders[0]: inputs_train[0],
                    placeholders[1]: inputs_train[1],
                    placeholders[2]: inputs_train[2],
                    placeholders[3]: inputs_train[3],
                    placeholders[4]: inputs_train[4]
                }

                eval_loss, _, eval_summary, eval_accuracy, test_logits = sess.run([graph_loss, graph_optmiser, graph_summary,
                                                                      graph_accuracy, graph_prediction], feed_dict=inputs_train_dict)

                if iteration_number % self.validate_after == 0:
                    inputs_validation = sess.run(list(inputs_validation_feed))
                    inputs_validation_dict = {
                        placeholders[0]: inputs_validation[0],
                        placeholders[1]: inputs_validation[1],
                        placeholders[2]: inputs_validation[2],
                        placeholders[3]: inputs_validation[3],
                        placeholders[4]: inputs_validation[4]
                    }

                    eval_loss_validation, eval_summary_validation, eval_accuracy_validation = sess.run([graph_loss, graph_summary_validation, graph_accuracy], feed_dict=inputs_validation_dict)
                    summary_writer.add_summary(eval_summary_validation, iteration_number)
                    print("Validation - Iteration %4d: loss %0.5f accuracy %03.3f" % (iteration_number, eval_loss_validation, eval_accuracy_validation))

                print("Training   - Iteration %4d: loss %0.5f accuracy %03.3f" % (iteration_number, eval_loss, eval_accuracy))
                iteration_number += 1
                summary_writer.add_summary(eval_summary, iteration_number)
                if iteration_number % self.save_after_iterations == 0:
                    print("\n\nINFO: Saving model\n\n")
                    self.saver_all.save(sess, self.model_path)
                    with open(self.model_path + '.txt', 'w') as f:
                        f.write(str(iteration_number))

            # # Stop the threads
            # coord.request_stop()
            #
            # # Wait for threads to stop
            # coord.join(threads)

