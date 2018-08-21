__author__ = 'indiano'

from sklearn.linear_model import LogisticRegression

# The Ensemble of the best models discovered through Auto-TPOT.

# LogisticRegression(input_matrix, C=20.0, dual=False, penalty=l1)
best_model = {'model': LogisticRegression(C=20.0, dual=False, penalty='l1')}
