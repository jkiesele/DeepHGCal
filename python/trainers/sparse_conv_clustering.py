import tensorflow as tf
import numpy as np
import os
import configparser as cp
from libs.helpers import get_num_parameters
from experiment.classification_model_test_result import ClassificationModelTestResult
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler.model_analyzer import Profiler
from models.sparse_conv_cluster_spatial_1 import SparseConvClusteringSpatial1
from models.sparse_conv_cluster_spatial_2_min_loss import SparseConvClusteringSpatialMinLoss
from readers import ReaderFactory


class SparseConvClusteringTrainer:
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
        self.learning_rate = float(self.config['learning_rate'])
        self.training_files = self.config['training_files_list']
        self.validation_files = self.config['validation_files_list']
        self.test_files = self.config['test_files_list']
        self.validate_after = int(self.config['validate_after'])

        self.num_batch = int(self.config['batch_size'])
        self.num_max_entries = int(self.config['max_entries'])
        self.num_data_dims = int(self.config['num_data_dims'])

        self.spatial_features_indices = tuple([int(x) for x in (self.config['input_spatial_features_indices']).split(',')])
        self.spatial_features_local_indices = tuple([int(x) for x in (self.config['input_spatial_features_local_indices']).split(',')])
        self.other_features_indices = tuple([int(x) for x in (self.config['input_other_features_indices']).split(',')])
        self.target_indices = tuple([int(x) for x in (self.config['target_indices']).split(',')])
        self.reader_type = self.config['reader_type'] if len(self.config['reader_type']) != 0 else "data_and_num_entries_reader"

        self.reader_factory = ReaderFactory()


    def initialize(self):
        model_type = self.config['model_type']
        self.model = globals()[model_type](
            len(self.spatial_features_indices),
            len(self.spatial_features_local_indices),
            len(self.other_features_indices),
            len(self.target_indices),
            self.num_batch,
            self.num_max_entries,
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

    def profile(self):
        raise ("Not implemented error")


    def train(self):
        self.initialize()
        print("Beginning to train network with parameters", get_num_parameters(self.model.get_variable_scope()))
        placeholders = self. model.get_placeholders()
        graph_loss = self.model.get_losses()
        graph_optmiser = self.model.get_optimizer()
        graph_summary = self.model.get_summary()
        graph_summary_validation = self.model.get_summary_validation()
        graph_output = self.model.get_compute_graphs()
        graph_temp = self.model.get_temp()

        if self.from_scratch:
            self.clean_summary_dir()

        inputs_feed = self.reader_factory.get_class(self.reader_type)(self.training_files, self.num_max_entries, self.num_data_dims, self.num_batch).get_feeds()
        inputs_validation_feed = self.reader_factory.get_class(self.reader_type)(self.validation_files, self.num_max_entries, self.num_data_dims, self.num_batch).get_feeds()

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
                    placeholders[0]: inputs_train[0][:, :, self.spatial_features_indices],
                    placeholders[1]: inputs_train[0][:, :, self.spatial_features_local_indices],
                    placeholders[2]: inputs_train[0][:, :, self.other_features_indices],
                    placeholders[3]: inputs_train[0][:, :, self.target_indices],
                    placeholders[4]: inputs_train[1]
                }

                t, eval_loss, _, eval_summary, eval_output = sess.run([graph_temp, graph_loss, graph_optmiser, graph_summary,
                                                                      graph_output], feed_dict=inputs_train_dict)

                if iteration_number % self.validate_after == 0:
                    inputs_validation = sess.run(list(inputs_validation_feed))
                    inputs_validation_dict = {
                        placeholders[0]: inputs_validation[0][:, :, self.spatial_features_indices],
                        placeholders[1]: inputs_validation[0][:, :, self.spatial_features_local_indices],
                        placeholders[2]: inputs_validation[0][:, :, self.other_features_indices],
                        placeholders[3]: inputs_validation[0][:, :, self.target_indices],
                        placeholders[4]: inputs_validation[1]
                    }

                    eval_loss_validation, eval_summary_validation= sess.run([graph_loss, graph_summary_validation], feed_dict=inputs_validation_dict)
                    summary_writer.add_summary(eval_summary_validation, iteration_number)
                    print("Validation - Iteration %4d: loss %0.5f" % (iteration_number, eval_loss_validation))

                print("Training   - Iteration %4d: loss %0.5f" % (iteration_number, eval_loss))
                print(t[0])
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
        raise ("Not implemented error")