__author__ = 'indiano'

import argparse
import os
import timeit

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.data_utils import DataUtils
from utils.plot import Plot

'''
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''

'''
Training:

python run.py --root_dir "./" --data_dir "../../data/keras" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/keras" --save_dir "../../save/keras" --test_size 33 

Sampling:

python run.py --root_dir "./" --data_dir "../../data/keras" --hdf_file "../../data/hdf/sportswear" 
--checkpoint_dir "../../checkpoint/keras" --save_dir "../../save/keras" --test_size 33 --run_type sample
'''

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data and model checkpoints directories
parser.add_argument('--root_dir', type=str, default='./',
                    help='root directory of the project')
parser.add_argument('--data_dir', type=str, default='../../data/keras',
                    help="""name of raw events folder if the hdf file not generated or
                    data directory containing input with training examples.""")
parser.add_argument('--hdf_file', type=str, default='../../data/hdf/sportswear',
                    help='stored or new hdf filename without .hdf extension.')
parser.add_argument('--checkpoint_dir', type=str, default='../../checkpoint/keras',
                    help='directory to store checkpointed models.')
parser.add_argument('--save_dir', type=str, default='../../save/keras',
                    help='directory to store graphs & plots')
parser.add_argument('--run_type', type=str, default='train',
                    help='train or sample for training or sampling.')

# Optimization
parser.add_argument('--test_size', type=int, default=33,
                    help="""% of total data equals the test size for train test split.
                     Please enter a value between 0-100.""")

# Parsed/collected all the arguments
args = parser.parse_args()


class Modelkeras:

    def __init__(self, args):
        self.args = args

        # Loading data
        self.data = pd.read_hdf(self.args.hdf_file + '.hdf')
        self.data.drop_duplicates(subset=['url'], inplace=True)

        self.max_features = 20000
        self.batch_size = 32

        # Register `pandas.progress_apply` and `pandas.Series.map_apply` with `tqdm`;
        # (can use `tqdm_gui`, `tqdm_notebook`, optional kwargs, etc.)
        tqdm.pandas(desc="my bar")
        # cut texts after this number of words (among top max_features most common words)
        self.maxlen = max(self.data['url'].progress_map(lambda x: len(x))) + 1

        # Fix the seed
        self.seed = 21
        np.random.seed(self.seed)

        # Splitting data into train, test & validation sets
        self.x_train, self.x_val_test, self.y_train, self.y_val_test = train_test_split(self.data['url'].values,
                                                                                        self.data['label'],
                                                                                        test_size=.33,
                                                                                        random_state=self.seed,
                                                                                        stratify=self.data['label'])

        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(self.x_val_test, self.y_val_test,
                                                                            test_size=.5, random_state=self.seed,
                                                                            stratify=self.y_val_test)

        print('\n*************** Data statistics ****************')
        datautils = DataUtils(self.args)
        datautils.data_Stats(self.y_train, self.y_val, self.y_test)

    def train(self, embedding_type):
        # Create the tokenizer & fit on all the url texts
        self.tokenizer = Tokenizer(num_words=self.max_features, lower=True)
        self.tokenizer.fit_on_texts(self.x_train)

        # Dumping tokenizer using joblib which is faster than pickle
        joblib.dump(self.tokenizer, os.path.join(self.args.data_dir, '{}/tokenizer.pickle'.format(embedding_type)))

        # Printing the learnt summary about url texts
        print('Tokenizer summary after learning on the url texts.')
        print('word_counts: {}'.format(self.tokenizer.word_counts))
        print('document_count: {}'.format(self.tokenizer.document_count))
        print('word_index: {}'.format(self.tokenizer.word_index))
        print('word_docs: {}'.format(self.tokenizer.word_docs))

        # Generating sequences & padding for efficient training of our neural network
        # Transforms each text in texts to a sequence of integers.
        self.train_sequences = self.tokenizer.texts_to_sequences(self.x_train)
        self.train_padded_sequences = sequence.pad_sequences(self.train_sequences, maxlen=self.maxlen)

        self.val_sequences = self.tokenizer.texts_to_sequences(self.x_val)
        self.val_padded_sequences = sequence.pad_sequences(self.val_sequences, maxlen=self.maxlen)

        self.test_sequences = self.tokenizer.texts_to_sequences(self.x_test)
        self.test_padded_sequences = sequence.pad_sequences(self.test_sequences, maxlen=self.maxlen)

        # Setting checkpoint & early stopping
        checkpoint_path = os.path.join(os.path.join(self.args.checkpoint_dir, embedding_type),
                                       'model_best_weights.{epoch:02d}-{val_acc:.4f}.hdf5')
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')
        callbacks_list = [checkpoint, early_stop]

        # Running the pipeline
        # Fixing the seed again
        np.random.seed(self.seed)

        print('\n\nBuild model...')
        model = Sequential()
        model.add(Embedding(self.max_features, 200))
        model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('Training starts...')
        model.fit(self.train_padded_sequences, self.y_train,
                  batch_size=self.batch_size,
                  epochs=15,
                  validation_data=(self.val_padded_sequences, self.y_val),
                  verbose=2, callbacks=callbacks_list)

        score, acc = model.evaluate(self.test_padded_sequences, self.y_test,
                                    batch_size=self.batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

    def sample(self, embedding_type):
        # Dumping tokenizer using joblib which is faster than pickle
        tokenizer = joblib.load(os.path.join(self.args.data_dir, '{}/tokenizer.pickle'.format(embedding_type)))
        test_sequences = tokenizer.texts_to_sequences(self.x_test)
        test_padded_sequences = sequence.pad_sequences(test_sequences, maxlen=self.maxlen)

        # Setting checkpoint & early stopping
        checkpoint_path = os.path.join(os.path.join(self.args.checkpoint_dir, embedding_type),
                                       'model_best_weights.10-0.9961.hdf5')

        print('\n\nLoading the best model...')
        model = load_model(checkpoint_path)

        # Calculating time per prediction
        # Start time ******************************************************************************
        start = timeit.default_timer()

        print('\n\nPrediction starts...')
        predictions = model.predict_classes(test_padded_sequences)

        end = timeit.default_timer()
        # End Time *******
        print('Time per prediction : {}'.format((end - start) / test_padded_sequences.shape[0]))

        self.visualize(predictions, model)

    def visualize(self, predictions, model, save_dir='../../save/keras', plt_name='keras'):
        # Evaluate predictions using accuracy metrics
        accuracy = accuracy_score(self.y_test, predictions)
        print('{} Classification'.format(model))
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
        plot.plotly(cnf_matrix, classification_report, os.path.join(self.args.save_dir, embedding_type), plt_name)


if __name__ == '__main__':
    embedding_type = 'embed_tokenizer'
    # Initializing & running keras model
    model = Modelkeras(args)

    if args.run_type == 'sample':
        print('\n\n****************** Sampling started on test data. *******************\n\n')
        model.sample(embedding_type)
        print('\n\n****************** Classification done. Enjoy Life. :) *******************')
    else:
        print('\n\n****************** Keras model training started. *******************\n\n')
        model.train(embedding_type)
        print('Best model will be saved in {}'.format(os.path.join(args.checkpoint_dir, embedding_type)))
        print('\n\n****************** Done Done Done. Enjoy Life. :) *******************')
