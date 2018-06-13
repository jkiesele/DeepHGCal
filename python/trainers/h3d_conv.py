import tensorflow as tf
from models.hgcal_conv_conv_3d import HgCal3d
from models.hgcal_conv_3d_2 import HgCal3d2
import numpy as np
import os
import configparser as cp
import sys
import pickle
import gzip
from helpers.helpers import get_num_parameters
from experiment.classification_model_test_result import ClassificationModelTestResult
import matplotlib.pyplot as plt

class H3dConvTrainer:
    def read_config(self, config_file_path, config_name):
        config_file = cp.ConfigParser()
        config_file.read(config_file_path)
        self.config = config_file[config_name]

    def __init__(self, config_file, config_name):
        self.read_config(config_file, config_name)

        self.from_scratch = int(self.config['from_scratch'])==1
        self.model_path = self.config['model_path']
        self.summary_path = self.config['summary_path']
        self.test_out_path = self.config['test_out_path']
        self.train_for_iterations = int(self.config['train_for_iterations'])
        self.save_after_iterations = int(self.config['save_after_iterations'])

        self.num_batch = int(self.config['batch_size'])

        self.num_dim_x = int(self.config['input_dim_x'])
        self.num_dim_y = int(self.config['input_dim_y'])
        self.num_dim_z = int(self.config['input_dim_z'])
        self.num_input_features = int(self.config['input_features'])

        self.num_classes = int(self.config['num_classes'])
        self.learning_rate = float(self.config['learning_rate'])
        self.training_files = self.config['training_files_list']
        self.validation_files = self.config['validation_files_list']
        self.test_files = self.config['test_files_list']
        self.validate_after = int(self.config['validate_after'])

    def initialize(self):
        model_type = self.config['model_type']
        self.model = globals()[model_type](
            self.num_dim_x,
            self.num_dim_y,
            self.num_dim_z,
            self.num_input_features,
            self.num_batch,
            self.num_classes,
            self.learning_rate
        )

        self.model.initialize()
        self.saver_sparse = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.model.get_variable_scope()))

    def clean_summary_dir(self):
        print("Cleaning summary dir")
        for the_file in os.listdir(self.summary_path):
            file_path = os.path.join(self.summary_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def __get_input_feeds(self, files_list, repeat=True, shuffle_size=None):
        def _parse_function(example_proto):
            keys_to_features = {
                'x': tf.FixedLenFeature((self.num_dim_x, self.num_dim_y, self.num_dim_z, self.num_input_features), tf.float32),
                'labels_one_hot': tf.FixedLenFeature((self.num_classes,), tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, keys_to_features)
            return parsed_features['x'], parsed_features['labels_one_hot']

        with open(files_list) as f:
            content = f.readlines()
        file_paths = [x.strip() for x in content]
        dataset = tf.data.TFRecordDataset(file_paths, compression_type='GZIP')
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(buffer_size=self.num_batch * 3 if shuffle_size is None else shuffle_size)
        dataset = dataset.repeat(None if repeat else 1)
        dataset = dataset.batch(self.num_batch)
        iterator = dataset.make_one_shot_iterator()
        inputs = iterator.get_next()

        return inputs

    def train(self):
        self.initialize()
        print("Beginning to train network with parameters", get_num_parameters(self.model.get_variable_scope()))

        placeholders = self. model.get_placeholders()
        graph_loss = self.model.get_losses()
        graph_optmiser = self.model.get_optimizer()
        graph_summary = self.model.get_summary()
        graph_summary_validation = self.model.get_summary_validation()
        graph_accuracy = self.model.get_accuracy()
        graph_logits, graph_prediction = self.model.get_compute_graphs()
        graph_temp = self.model.get_temp()

        if self.from_scratch:
            self.clean_summary_dir()

        inputs_feed = self.__get_input_feeds(self.training_files)
        inputs_validation_feed = self.__get_input_feeds(self.validation_files)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        with tf.Session() as sess:
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

            if not self.from_scratch:
                self.saver_sparse.restore(sess, self.model_path)
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
                    placeholders[1]: inputs_train[1]
                }

                t, eval_loss, _, eval_summary, eval_accuracy, test_logits = sess.run([graph_temp, graph_loss, graph_optmiser, graph_summary,
                                                                      graph_accuracy, graph_prediction], feed_dict=inputs_train_dict)

                if iteration_number % self.validate_after == 0:
                    inputs_validation = sess.run(list(inputs_validation_feed))
                    inputs_validation_dict = {
                        placeholders[0]: inputs_validation[0],
                        placeholders[1]: inputs_validation[1]
                    }

                    eval_loss_validation, eval_summary_validation, eval_accuracy_validation = sess.run([graph_loss, graph_summary_validation, graph_accuracy], feed_dict=inputs_validation_dict)
                    summary_writer.add_summary(eval_summary_validation, iteration_number)
                    print("Validation - Iteration %4d: loss %0.5f accuracy %03.3f" % (iteration_number, eval_loss_validation, eval_accuracy_validation))

                print("Training   - Iteration %4d: loss %0.5f accuracy %03.3f" % (iteration_number, eval_loss, eval_accuracy))
                print(t[0])

                # print(inputs_train[3][0])
                iteration_number += 1
                summary_writer.add_summary(eval_summary, iteration_number)
                if iteration_number % self.save_after_iterations == 0:
                    print("\n\nINFO: Saving model\n\n")
                    self.saver_sparse.save(sess, self.model_path)
                    with open(self.model_path + '.txt', 'w') as f:
                        f.write(str(iteration_number))

            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)

    def test(self):
        self.num_batch = 1
        self.initialize()
        print("Beginning to test network with parameters", get_num_parameters(self.model.get_variable_scope()))

        placeholders = self.model.get_placeholders()
        graph_loss = self.model.get_losses()
        graph_optmiser = self.model.get_optimizer()
        graph_summary = self.model.get_summary()
        graph_summary_validation = self.model.get_summary_validation()
        graph_confusion_matrix = self.model.get_confusion_matrix()
        graph_accuracy = self.model.get_accuracy()
        graph_logits, graph_prediction = self.model.get_compute_graphs()
        graph_temp = self.model.get_temp()

        inputs_feed = self.__get_input_feeds(self.test_files, repeat=False)

        accuracy_sum = 0
        num_examples = 0

        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        with tf.Session() as sess:
            sess.run(init)
            print("Beginning to test network with parameters", get_num_parameters(self.model.get_variable_scope()))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            self.saver_sparse.restore(sess, self.model_path)
            print("\n\nINFO: Loading model\n\n")
            iteration_number = 0

            labels = np.zeros((1000000, self.num_classes))
            scores = np.zeros((1000000, self.num_classes))

            print("Starting iterations")
            while iteration_number < 1000000:
                try:
                    inputs = sess.run(inputs_feed)
                except tf.errors.OutOfRangeError:
                    break

                inputs_train_dict = {
                    placeholders[0]: inputs[0],
                    placeholders[1]: inputs[1]
                }
                labels[iteration_number] = np.squeeze(inputs[1])

                t, eval_loss, eval_accuracy, eval_confusion, test_logits, eval_logits = sess.run(
                    [graph_temp, graph_loss, graph_accuracy, graph_confusion_matrix, graph_prediction, graph_logits],
                    feed_dict=inputs_train_dict)

                confusion_matrix += eval_confusion
                accuracy_sum += eval_accuracy * self.num_batch
                num_examples += self.num_batch

                scores[iteration_number] = np.squeeze(eval_logits)

                print("Test - Batch %4d: loss %0.5f accuracy %03.3f accuracy (cumm) %03.3f" % (
                iteration_number, eval_loss, eval_accuracy, accuracy_sum / num_examples))
                iteration_number += 1

            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)

        classes_names = 'Electron', 'Muon', 'Pion Charged', 'Pion Neutral', 'K0 Long', 'K0 Short' # TODO: Pick from config

        test_result = ClassificationModelTestResult()
        test_result.initialize(confusion_matrix, labels, scores, self.model.get_human_name(),
                               get_num_parameters(self.model.get_variable_scope()), classes_names, self.summary_path)
        test_result.evaluate(self.test_out_path)

        print("Evaluation complete")
        print("Evaluation accuracy ", accuracy_sum / num_examples)
        print("Confusion matrix:")
        print(confusion_matrix)

