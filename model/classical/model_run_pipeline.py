__author__ = 'indiano'

# Active wear dataset classification, binary labels

import os
import timeit

import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from utils.plot import Plot


class ModelRunPipeline:

    def __init__(self, args, pipeline, data):
        self.args = args
        self.pipeline = pipeline
        self.data = data

    def run_pipeline(self):
        """
        run_pipeline function runs the actual pipeline.
        :return:
        """

        # Train & Test data split using sklearn train_test_split module
        X_train, X_test, y_train, y_test = train_test_split(self.data['url'], self.data['label'], test_size=0.33,
                                                            random_state=21, stratify=self.data['label'])
        print("*******************\nTrain set : {} \n Test set : {}\n*******************\n".format(X_train.shape[0],
                                                                                                   X_test.shape[0]))

        # Running the pipeline
        model = self.pipeline.fit(X_train, y_train)

        print('Saving the {} model after fitting on training data.'.format(str(self.args.model).upper()))
        # Dumping tokenizer
        joblib.dump(model, os.path.join(self.args.checkpoint_dir, '{}.pickle'.format(self.args.model)))

        # Calculating time per prediction
        # Start time ******************************************************************************
        start = timeit.default_timer()

        # Predicting label, confidence probability on the test data set
        predictions = model.predict(X_test)
        predictions_prob = model.predict_proba(X_test)

        # Binary class values : rounding them to 0 or 1
        predictions = [round(value) for value in predictions]

        end = timeit.default_timer()
        # End Time ******************************************************************************

        print('Time per prediction : {}'.format((end - start) / X_test.shape[0]))

        # evaluate predictions using accuracy metrics
        accuracy = accuracy_score(y_test, predictions)
        print('{} Classification'.format(self.args.model))
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # evaluate predictions using confusion metrics and plot confusion matrix
        classification_report = metrics.classification_report(predictions, y_test,
                                                              target_names=['NadaSportswear', 'Sportswear'])
        print(classification_report)

        # Plotting confusion matrix
        cnf_matrix = confusion_matrix(y_test, predictions)
        np.set_printoptions(precision=2)

        # Plot module is used for plotting confusion matrix, classification report
        plot = Plot()
        plot.plotly(cnf_matrix, classification_report, self.args.save_dir, self.args.model)
