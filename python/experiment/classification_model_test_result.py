import pickle
import gzip
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from sklearn import metrics
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class ClassificationModelTestResult:
    def initialize(self, confusion_matrix, labels, logits, model_name, trainable_parameters, classes_names, summary_path):
        test_result = dict()
        test_result['confusion_matrix'] = confusion_matrix
        test_result['logits'] = logits
        test_result['labels'] = labels
        test_result['model_name'] = model_name
        test_result['trainable_parameters'] = trainable_parameters
        test_result['classes_names'] = classes_names

        self.test_result = test_result
        self.initialized = True
        self.pick_up_summary(summary_path)


    def pick_up_summary(self, summary_path):
        event_acc = EventAccumulator(summary_path)
        event_acc.Reload()
        # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
        _, steps_loss, vals_loss = zip(*event_acc.Scalars('Loss'))
        _, steps_accuracy, vals_accuracy = zip(*event_acc.Scalars('Accuracy'))
        _, steps_validation_loss, vals_validation_loss = zip(*event_acc.Scalars('Validation_Loss'))
        _, steps_validation_accuracy, vals_validation_accuracy = zip(*event_acc.Scalars('Validation_Accuracy'))

        training_history = dict()
        training_history['steps_loss'] = steps_loss
        training_history['steps_accuracy'] = steps_accuracy
        training_history['steps_validation_loss'] = steps_validation_loss
        training_history['steps_validation_accuracy'] = steps_validation_accuracy
        training_history['values_loss'] = vals_loss
        training_history['values_accuracy'] = vals_accuracy
        training_history['values_validation_loss'] = vals_validation_loss
        training_history['values_validation_accuracy'] = vals_validation_accuracy

        self.test_result['training_history'] = training_history

    def __init__(self):
        self.initialized = False

    def output_to_file(self, output_path):
        f = gzip.open(output_path, 'wb')
        pickle.dump(self.test_result, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

        print("Output written to ", output_path)

    def load_from_file(self, file_name):
        if self.initialized:
            print("Already initialized")
            return
        # restore the object
        f = gzip.open(file_name, 'rb')
        self.test_result = pickle.load(f)
        f.close()

    def get_confusion_matrix(self):
        return self.test_result['confusion_matrix']

    def get_logits(self):
        return self.test_result['logits']

    def get_labels(self):
        return self.test_result['lables']

    def get_model_name(self):
        return self.test_result['model_name']

    def get_trainable_parameters(self):
        return self.test_result['trainable_parameters']

    def output_rocs(self):
        # Output ROC curves
        labels = self.test_result['labels']
        logits = self.test_result['logits']

        classes_names = self.test_result['classes_names']
        num_classes = len(classes_names)
        shape_labels = np.shape(labels)
        shape_logits = np.shape(logits)
        assert len(shape_labels) == 2 and shape_labels[1] == num_classes
        assert len(shape_logits) == 2 and shape_logits[1] == num_classes

        labels_indexed = np.argmax(labels, axis=1)

        for i in range(num_classes):
            scores = logits[:, i]
            fpr, tpr, thresholds = metrics.roc_curve(labels_indexed, scores, pos_label=i)
            roc_auc = metrics.auc(fpr, tpr)

            fig = plt.figure()
            plt.plot(fpr, tpr, label='AUC = %.2f' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.05])
            plt.ylim([0.0, 1.05])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC curve - %s' % classes_names[i])
            fig.savefig(os.path.join(self.output_folder, 'roc_' + (classes_names[i].strip())) + '.png')  # save the figure to file

    def output_training_history(self):
        training_history = self.test_result['training_history']

        fig = plt.figure()

        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        plt.plot(training_history['steps_loss'], training_history['values_loss'], label='Loss')
        plt.plot(training_history['steps_validation_loss'], training_history['values_validation_loss'], label='Validation Loss')

        plt.title('Loss')
        fig.savefig(os.path.join(self.output_folder, 'loss.png')) # save the figure to file

        fig = plt.figure()

        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')

        plt.plot(training_history['steps_accuracy'], training_history['values_accuracy'], label='Accuracy')
        plt.plot(training_history['steps_validation_accuracy'], training_history['values_validation_accuracy'], label='Validation Accuracy')

        plt.title('Accuracy')
        fig.savefig(os.path.join(self.output_folder, 'accuracy.png')) # save the figure to file

    def evaluate(self, output_folder):
        self.output_folder= output_folder
        self.output_rocs()
        self.output_training_history()
