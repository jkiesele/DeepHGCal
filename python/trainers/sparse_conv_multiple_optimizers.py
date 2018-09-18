import tensorflow as tf
import numpy as np
from libs.helpers import get_num_parameters
from trainers.sparse_conv import SparseConvTrainer


class SparseConvTrainerMulti(SparseConvTrainer):
    def __init__(self, config_file, config_name):
        super(SparseConvTrainerMulti, self).__init__(config_file, config_name)
        self.switch_after = int(self.config['switch_after_iterations'])

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

        place_holder_control_switch = self.model.get_place_holder_switch_control()

        if self.from_scratch:
            self.clean_summary_dir()

        inputs_feed = self._get_input_feeds(self.training_files)
        inputs_validation_feed = self._get_input_feeds(self.validation_files)

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
                number = int(iteration_number / self.switch_after) % 2
                switch_value = np.zeros((2), dtype=np.int64)
                switch_value[number] = 1
                switch_value = [1, 1] # TODO: Remove this later

                inputs_train = sess.run(list(inputs_feed))

                inputs_train_dict = {
                    placeholders[0]: inputs_train[0],
                    placeholders[1]: inputs_train[1],
                    placeholders[2]: inputs_train[2],
                    placeholders[3]: inputs_train[3],
                    placeholders[4]: inputs_train[4],
                    place_holder_control_switch: switch_value
                }

                t, eval_loss, _, eval_summary, eval_accuracy, test_logits = sess.run([graph_temp, graph_loss, graph_optmiser, graph_summary,
                                                                      graph_accuracy, graph_prediction], feed_dict=inputs_train_dict)

                if iteration_number % self.validate_after == 0:
                    inputs_validation = sess.run(list(inputs_validation_feed))
                    inputs_validation_dict = {
                        placeholders[0]: inputs_validation[0],
                        placeholders[1]: inputs_validation[1],
                        placeholders[2]: inputs_validation[2],
                        placeholders[3]: inputs_validation[3],
                        placeholders[4]: inputs_validation[4],
                        place_holder_control_switch: switch_value
                    }

                    eval_loss_validation, eval_summary_validation, eval_accuracy_validation = sess.run([graph_loss, graph_summary_validation, graph_accuracy], feed_dict=inputs_validation_dict)
                    summary_writer.add_summary(eval_summary_validation, iteration_number)
                    print("Validation - Iteration %4d: loss %0.5f accuracy %03.3f" % (iteration_number, eval_loss_validation, eval_accuracy_validation))

                print("Training   - Iteration %4d: loss %0.5f accuracy %03.3f" % (iteration_number, eval_loss, eval_accuracy))
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