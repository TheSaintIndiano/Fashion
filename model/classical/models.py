__author__ = 'indiano'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier


class Models:

    def __init__(self, args):
        self.args = args

    def model_pipeline(self):
        """
        Pipelines contains placeholder for different run pipelines.
        :param self.args: self.args list
        :return:
        """

        # tf-idf vectorizer for sentence/topic/document modelling
        tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

        if self.args.model == 'xgb':
            # XGB classifier
            xgb = XGBClassifier()
            pipeline = Pipeline([
                ('tfidf', tfidf_vectorizer),
                ('clf', xgb)
            ])

        elif self.args.model == 'svc':
            # SVC classifier, C = 1.0  # SVM regularization parameter
            svc = SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True)
            pipeline = Pipeline([
                ('tfidf', tfidf_vectorizer),
                ('clf', svc)
            ])

        elif self.args.model == 'naivebayes':
            # Naive Bayes classifier
            naive_bayes = MultinomialNB()
            pipeline = Pipeline([
                ('tfidf', tfidf_vectorizer),
                ('clf', naive_bayes)
            ])

        elif self.args.model == 'knn':
            # K Nearest Neighbour classifier
            knn = KNeighborsClassifier()
            pipeline = Pipeline([
                ('tfidf', tfidf_vectorizer),
                ('clf', knn)
            ])

        else:
            raise Exception("model type not supported: {}".format(self.args.model))

        return pipeline, self.args.model
