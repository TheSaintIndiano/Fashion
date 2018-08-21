__author__ = 'indiano'

# Active wear dataset classification, binary labels

import argparse
import os
import pickle
import timeit

import autosklearn.classification
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from config import autosklearn_config
from utils.plot import Plot

# Import files with special fileor path names
# spec = importlib.util.spec_from_file_location("ensemble_choices", "../../config/config.py")
# config = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(config)

"""
auto-sklearn frees a machine learning user from algorithm selection and hyperparameter tuning. 
It leverages recent advantages in Bayesian optimization, meta-learning and ensemble construction.
"""

'''
Training:

python run.py --root_dir "./" --data_dir "../../data/sportswear/events" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/auto-sklearn" --save_dir "../../save/auto-sklearn" --test_size 33 

Sampling:

python run.py --root_dir "./" --data_dir "../../data/sportswear/events" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/auto-sklearn" --save_dir "../../save/auto-sklearn" --test_size 33 --run_type sample
'''

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data and model checkpoints directories
parser.add_argument('--root_dir', type=str, default='./',
                    help='root directory of the project')
parser.add_argument('--data_dir', type=str, default='../../data/sportswear/events',
                    help="""name of raw events folder if the hdf file not generated or
                    data directory containing input with training examples.""")
parser.add_argument('--hdf_file', type=str, default='../../data/hdf/sportswear',
                    help='stored or new hdf filename without .hdf extension.')
parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoint/auto-sklearn',
                    help='directory to store checkpointed models.')
parser.add_argument('--save_dir', type=str, default='../../save/auto-sklearn',
                    help='directory to store graphs & plots')
parser.add_argument('--run_type', type=str, default='train',
                    help='train or sample for training or sampling.')

# Optimization
parser.add_argument('--test_size', type=int, default=33,
                    help="""% of total data equals the test size for train test split.
                     Please enter a value between 0-100.""")

# Parsed/collected all the arguments
args = parser.parse_args()


class AutoSklearn:

    def __init__(self, args):

        self.args = args

        # Loading data
        self.data = pd.read_hdf(self.args.hdf_file + '.hdf')
        # Categorical input features eg. categorical_features = ['url'] or ['color', 'size', 'url']
        self.data = self.data.loc[:, ['url', 'label']]
        self.data.drop_duplicates(subset=['url'], inplace=True)

        # Train & Test data split using sklearn train_test_split module
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['url'], self.data['label'],
                                                                                test_size=(self.args.test_size / 100),
                                                                                random_state=21,
                                                                                stratify=self.data['label'])
        print(
            "*******************\nTrain set : {} \n Test set : {}\n*******************\n".format(self.X_train.shape[0],
                                                                                                 self.X_test.shape[0]))

        # tf-idf vectorizer for sentence/topic/document modelling
        tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        self.train = tfidf_vectorizer.fit_transform(self.X_train)
        self.test = tfidf_vectorizer.transform(self.X_test)

    def run_autosklearn(self):

        # Running the autosklearn pipeline
        automl = autosklearn.classification.AutoSklearnClassifier()
        automl.fit(self.train, self.y_train)

        best_models = automl.show_models()
        print('The best models discovered through Auto-Sklearn is {}'.format(best_models))

        print('Saving the best model discovered through auto-sklearn.')
        # Dumping ensemble of the models
        with open(os.path.join(self.args.checkpoint_dir, 'auto-sklearn.pickle'), 'wb') as file:
            pickle.dump(automl, file, protocol=pickle.HIGHEST_PROTOCOL)

        # Calculating time per prediction
        # Start time ******************************************************************************
        start = timeit.default_timer()

        # Predicting label, confidence probability on the test data set
        predictions = automl.predict(self.test)
        predictions_prob = automl.predict_proba(self.test)

        # Binary class values : rounding them to 0 or 1
        predictions = [round(value) for value in predictions]

        end = timeit.default_timer()
        # End Time ******************************************************************************
        print('Time per prediction : {}'.format((end - start) / self.test.shape[0]))
        self.visualize(predictions, automl)

    def visualize(self, predictions, automl, save_dir='../../save/auto-sklearn', plt_name='auto-sklearn'):
        # Evaluate predictions using accuracy metrics
        accuracy = accuracy_score(self.y_test, predictions)
        print('{} Classification'.format(automl))
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # Evaluate predictions using confusion metrics and plot confusion matrix
        classification_report = metrics.classification_report(predictions, self.y_test,
                                                              target_names=['NadaSportswear', 'Sportswear'])
        print(classification_report)

        # Calculating confusion matrix
        cnf_matrix = confusion_matrix(self.y_test, predictions)
        np.set_printoptions(precision=2)

        # Plot module is used for plotting confusion matrix, classification report
        plot = Plot()
        plot.plotly(cnf_matrix, classification_report, save_dir, plt_name)

    def fit_and_predict(self, estimator, weight, X_train, y_train, X_test):
        '''
        Ensemble models being refitted based on the best saved configuration
        :param weight:
        :param X_train:
        :param y_train:
        :param X_test:
        :return:
        '''

        try:
            estimator.fit(X_train.copy(), y_train.copy())
            # Ensemble model being weighted
            predictions = estimator.predict(X_test.copy()) * weight
            predictions_prob = estimator.predict_proba(X_test.copy()) * weight

        except Exception as e:
            print(e)
            print(estimator.configuration)
            predictions = None
            predictions_prob = None

        return predictions, predictions_prob

    def sample(self):
        '''
        Sampling using the best ensemble models
        :return:
        '''

        print('Loading the best model discovered through auto-sklearn.')
        # Dumping tokenize
        with open(os.path.join(self.args.checkpoint_dir, 'auto-sklearn.pickle'), 'rb') as file:
            automl = pickle.load(file)

            print('The best models discovered through Auto-Sklearn is {}'.format(autosklearn_config.ensemble_choices))

        # Predicting label, confidence probability on the test data set
        # Running the pipeline
        all_predictions = Parallel(n_jobs=-1)(
            delayed(self.fit_and_predict)(estimator, weight, self.train, self.y_train, self.test) for weight, estimator
            in
            autosklearn_config.ensemble_choices)

        predictions = []
        predictions_prob = []

        for p, pp in all_predictions:
            predictions.append(p)
            predictions_prob.append(pp)

        # Visualize using prediction scores
        # Scores addition from different Ensemble models
        predictions = np.array(predictions)
        predictions = np.sum(predictions, axis=0).astype(np.float32)
        predictions = predictions.reshape((-1, 1))
        # Binary class values : rounding them to 0 or 1
        predictions = np.array([np.round(value) for value in predictions])

        self.visualize(predictions, automl, save_dir=self.args.save_dir)

        # Visualize using predictions probability
        # Scores addition from different Ensemble models
        predictions_prob = np.array(predictions_prob)
        predictions_prob = np.sum(predictions_prob, axis=0).astype(np.float32)
        predictions_prob = predictions_prob[:, 1].reshape((-1, 1))
        # Binary class values : rounding them to 0 or 1
        predictions_prob = np.array([np.round(value) for value in predictions_prob])

        self.visualize(predictions_prob, automl, save_dir=self.args.save_dir, plt_name='auto-sklearn_prob')


if __name__ == '__main__':

    # Initializing & running auto sklearn
    autosk = AutoSklearn(args)

    if args.run_type == 'sample':
        print('\n\n****************** Auto Sklearn sampling started. *******************\n\n')
        autosk.sample()
        print('\n\n****************** Sampling done. Enjoy Life. :) *******************')
    else:
        print('\n\n****************** Auto Sklearn training started. *******************\n\n')
        autosk.run_autosklearn()
        print('\n\n****************** Ensembling done. Enjoy Life. :) *******************')
