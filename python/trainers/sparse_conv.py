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


    def get_tfrecords_input_feeds(self, files_list):
        with open(files_list) as f:
            content = f.readlines()
        file_paths = [x.strip() for x in content]
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        reader = tf.TFRecordReader(options=options)
        filename_queue = tf.train.string_input_producer(file_paths)
        _, serialized_example = reader.read(filename_queue)

        feature = {'space_features': tf.FixedLenFeature([], tf.string),
                   'space_features_shape': tf.FixedLenFeature([], tf.string),
                   'all_features': tf.FixedLenFeature([], tf.string),
                   'all_features_shape': tf.FixedLenFeature([], tf.string),
                   'neighbor_matrix': tf.FixedLenFeature([], tf.string),
                   'neighbor_matrix_shape': tf.FixedLenFeature([], tf.string),
                   'labels_one_hot': tf.FixedLenFeature([], tf.string),
                   'labels_one_hot_shape': tf.FixedLenFeature([], tf.string),
                   'num_entries': tf.FixedLenFeature([], tf.string),
                   'num_entries_shape': tf.FixedLenFeature([], tf.string)} # Shouldn't need to store shape of num_entires

        # Decode the record read by the reader
        example = tf.parse_single_example(serialized_example, features=feature)
        space_features = example['space_features']
        space_features_shape = example['space_features_shape']
        all_features = example['all_features']
        all_features_shape = example['all_features_shape']
        neighbor_matrix = example['neighbor_matrix']
        neighbor_matrix_shape = example['neighbor_matrix_shape']
        labels_one_hot = example['labels_one_hot']
        labels_one_hot_shape = example['labels_one_hot_shape']
        num_entries = example['num_entries']
        num_entries_shape = example['num_entries_shape']

        space_features, space_features_shape, all_features, all_features_shape , neighbor_matrix , \
        neighbor_matrix_shape , labels_one_hot, labels_one_hot_shape, num_entries,num_entries_shape\
            = tf.train.shuffle_batch (
                [
                    space_features, space_features_shape, all_features, all_features_shape , neighbor_matrix , \
                    neighbor_matrix_shape , labels_one_hot, labels_one_hot_shape, num_entries,num_entries_shape
                ],
                batch_size=int(self.config['batch_size']), capacity=int(self.config['batch_size'])*10,
                min_after_dequeue=int(self.config['batch_size'])
            )

        feeds = {
            'space_features' : space_features,
            'space_features_shape' : space_features_shape,
            'all_features' : all_features,
            'all_features_shape' : all_features_shape,
            'neighbor_matrix' : neighbor_matrix,
            'neighbor_matrix_shape' : neighbor_matrix_shape,
            'labels_one_hot' : labels_one_hot,
            'labels_one_hot_shape' : labels_one_hot_shape,
            'num_entries' : num_entries,
            'num_entries_shape' : num_entries_shape
        }

        return feeds

    def __get_inputs(self, sess, input_feeds):
        space_features, space_features_shape, all_features, all_features_shape, neighbor_matrix, \
        neighbor_matrix_shape, labels_one_hot, labels_one_hot_shape, num_entries, num_entries_shape = sess.run([
            input_feeds['space_features'],
            input_feeds['space_features_shape'],
            input_feeds['all_features'],
            input_feeds['all_features_shape'],
            input_feeds['neighbor_matrix'],
            input_feeds['neighbor_matrix_shape'],
            input_feeds['labels_one_hot'],
            input_feeds['labels_one_hot_shape'],
            input_feeds['num_entries'],
            input_feeds['num_entries_shape']
        ])


        space_features = [np.frombuffer(i, dtype=np.float32).reshape(self.num_max_entries, self.num_spatial_features) for i in space_features]
        all_features = [np.frombuffer(i, dtype=np.float32).reshape(self.num_max_entries, self.num_all_features) for i in all_features]
        neighbor_matrix = [np.frombuffer(i, dtype=np.int32).reshape(self.num_max_entries, self.num_max_neighbors) for i in neighbor_matrix]
        labels_one_hot = [np.frombuffer(i, dtype=np.int32).reshape(self.num_classes) for i in labels_one_hot] # Actually doesn't need to be resized but whatever
        num_entries = [int(np.frombuffer(i, dtype=np.int32).reshape(1)) for i in num_entries]

        return space_features, all_features, neighbor_matrix, labels_one_hot, num_entries

    def train(self):
        placeholder_space_features, placeholder_all_features, placeholder_neighbors_matrix, \
        placeholder_labels, placeholder_num_entries = self. model.get_placeholders()
        graph_loss = self.model.get_losses()
        graph_optmiser = self.model.get_optimizer()
        graph_summary = self.model.get_summary()
        graph_summary_validation = self.model.get_summary_validation()
        graph_accuracy = self.model.get_accuracy()
        graph_logits, graph_prediction = self.model.get_compute_graphs()


        if self.from_scratch:
            self.clean_summary_dir()

        input_feeds = self.get_tfrecords_input_feeds(self.training_files)
        input_feeds_validation = self.get_tfrecords_input_feeds(self.validation_files)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        with tf.Session() as sess:
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

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
                inputs = self.__get_inputs(sess, input_feeds)
                inputs_validation = self.__get_inputs(sess, input_feeds)

                input_dict = {
                    placeholder_space_features: inputs[0],
                    placeholder_all_features: inputs[1],
                    placeholder_neighbors_matrix: inputs[2],
                    placeholder_labels: inputs[3],
                    placeholder_num_entries: inputs[4]
                }

                inputs_validation_dict = {
                    placeholder_space_features: inputs_validation[0],
                    placeholder_all_features: inputs_validation[1],
                    placeholder_neighbors_matrix: inputs_validation[2],
                    placeholder_labels: inputs_validation[3],
                    placeholder_num_entries: inputs_validation[4]
                }

                eval_loss, _, eval_summary, eval_accuracy, test_logits = sess.run([graph_loss, graph_optmiser, graph_summary,
                                                                      graph_accuracy, graph_prediction], feed_dict=input_dict)

                if iteration_number % self.validate_after == 0:
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


            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)

