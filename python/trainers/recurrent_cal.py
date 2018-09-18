import tensorflow as tf
import configparser as cp
from ..models.recurrent_cal import RecurrentCal
import numpy as np
import os
from DeepJetCore.DataCollection import DataCollection


class RecurrentCalTrainer:
    def read_config(self, config_file_path, config_name):
        config_file = cp.ConfigParser()
        config_file.read(config_file_path)
        self.config = config_file[config_name]

    def __init__(self, config_file, config_name):
        self.read_config(config_file, config_name)
        self.model = RecurrentCal(int(self.config['batch_size']), int(self.config['num_layers']), int(self.config['num_channels']))
        self.model.initialize()
        self.saver_all = tf.train.Saver() # TODO: Might want to move variables etc to a scope or something?
        self.from_scratch = int(self.config['from_scratch'])==1
        self.model_path = self.config['model_path']
        self.summary_path = self.config['summary_path']
        self.train_for_iterations = int(self.config['train_for_iterations'])
        self.save_after_iterations = int(self.config['save_after_iterations'])
        self.use_tf_records = int(self.config['use_tf_records'])==1
        self.batch_size = int(self.config['batch_size'])

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

    def get_record_placeholders(self):
        with open(self.config['training_files_list']) as f:
            content = f.readlines()
        file_paths = [x.strip() for x in content]
        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        reader = tf.TFRecordReader(options=options)
        filename_queue = tf.train.string_input_producer(file_paths)
        _, serialized_example = reader.read(filename_queue)

        feature = {'x': tf.FixedLenFeature([], tf.string),
                   'y': tf.FixedLenFeature([], tf.string)}
        # Decode the record read by the reader
        example = tf.parse_single_example(serialized_example, features=feature)
        record_input = example['x']
        record_target = example['y']
        record_batch_input, record_batch_target = tf.train.shuffle_batch(
            [record_input, record_target], batch_size=int(self.config['batch_size']), capacity=int(self.config['batch_size'])*10,
            min_after_dequeue=int(self.config['batch_size']))

        return record_batch_input, record_batch_target


    def train(self):

        placeholder_input, placeholder_output = self.model.get_placeholders()
        graph_output = self.model.get_compute_graphs()
        graph_loss = self.model.get_losses()
        graph_optmiser = self.model.get_optimizer()
        graph_summary = self.model.get_summary()


        if self.from_scratch:
            self.clean_summary_dir()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            if self.use_tf_records:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                record_batch_input, record_batch_target = self.get_record_placeholders()
            else:
                input_data = self.config['train_data_path']
                train_data = DataCollection()
                train_data.readFromFile(input_data)

                val_data = train_data.split(0.1)
                train_data = train_data.split(0.9)
                train_data.setBatchSize(self.batch_size)
                val_data.setBatchSize(self.batch_size)
                val_data_generator = train_data.generator()
                train_data_generator = train_data.generator()

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
                if self.use_tf_records:
                    input, output = sess.run([record_batch_input, record_batch_target])
                    input = [np.fromstring(''.join(i)).reshape(13, 13, int(self.config['num_layers']),
                                                               int(self.config['num_channels'])) for i in input]
                    output = [np.fromstring(''.join(i)).reshape(13, 13, int(self.config['num_layers'])) for i in output]
                else:
                    input, output,_ = train_data_generator.next()
                    input = np.squeeze(input,axis=0)
                    output = np.squeeze(output,axis=0)

                _, eval_loss, _ , eval_summary= sess.run([graph_output, graph_loss, graph_optmiser, graph_summary],
                                      feed_dict={placeholder_input: input, placeholder_output: output})
                print("Iteration %4d: loss %0.5f" % (iteration_number, eval_loss))
                iteration_number += 1
                summary_writer.add_summary(eval_summary, iteration_number)
                if iteration_number % self.save_after_iterations == 0:
                    print("\n\nINFO: Saving model\n\n")
                    self.saver_all.save(sess, self.model_path)
                    with open(self.model_path + '.txt', 'w') as f:
                        f.write(str(iteration_number))
            if self.use_tf_records:
                # Stop the threads
                coord.request_stop()

                # Wait for threads to stop
                coord.join(threads)