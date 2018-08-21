__author__ = 'indiano'

# Active wear dataset classification, binary labels

import argparse
import os
import timeit

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

from utils.plot import Plot

# Import files with special fileor path names
# spec = importlib.util.spec_from_file_location("ensemble_choices", "../../config/config.py")
# config = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(config)

"""
tpot frees a machine learning user from algorithm selection and hyperparameter tuning. 
It leverages recent advantages in Genetic programming.

In artificial intelligence, genetic programming (GP) is a technique whereby 
computer programs are encoded as a set of genes that are then modified (evolved) using 
an evolutionary algorithm (often a genetic algorithm, "GA") – it is an application of 
(for example) genetic algorithms where the space of solutions consists of computer programs. 
The results are computer programs that are able to perform well in a predefined task. 
The methods used to encode a computer program in an artificial chromosome and to evaluate 
its fitness with respect to the predefined task are central in the GP technique and still the subject of active research.
"""

'''
Training:

python run.py --root_dir "./" --data_dir "../../data/sportswear/events" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/auto-tpot" --save_dir "../../save/auto-tpot" --test_size 33 
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
parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoint/auto-tpot',
                    help='directory to store checkpointed models.')
parser.add_argument('--save_dir', type=str, default='../../save/auto-tpot',
                    help='directory to store graphs & plots')
parser.add_argument('--run_type', type=str, default='train',
                    help='train input word for training.')

# Optimization
parser.add_argument('--test_size', type=int, default=33,
                    help="""% of total data equals the test size for train test split.
                     Please enter a value between 0-100.""")

# Parsed/collected all the arguments
args = parser.parse_args()


class AutoTpot:

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

    def run_AutoTpot(self):
        # Running the AutoTpot pipeline
        automl = TPOTClassifier(generations=1, verbosity=2, config_dict='TPOT sparse')
        automl.fit(self.train, self.y_train)

        # TPOT produces ready-to-run, standalone Python code for the best-performing model,
        # in the form of a scikit-learn pipeline.
        # Exporting the best models
        automl.export(os.path.join(self.args.save_dir, 'tpot-sportswear.py'))

        print('The best pipeline discovered through auto-tpot is {}'.format(automl.fitted_pipeline_))

        print('Saving the best model discovered through TPOT.')
        # Dumping ensemble of the models
        joblib.dump(automl, os.path.join(self.args.checkpoint_dir, 'auto-tpot.pickle'))

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

    def visualize(self, predictions, automl, save_dir='../../save/auto-tpot', plt_name='auto-tpot'):
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


if __name__ == '__main__':
    # Initializing & running auto tpot
    autosk = AutoTpot(args)

    print('\n\n****************** Auto TPOT training started. *******************\n\n')
    autosk.run_AutoTpot()
    print('\n\n****************** Ensembling done. Enjoy Life. :) *******************')
