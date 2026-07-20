"""
Shared BaggedSVM class.
Used by both week3_step1c_svm_bagged_ensemble.py (training) and
week4_step1_evaluate_all_models.py (evaluation).
"""

import numpy as np


class BaggedSVM:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probs = np.stack([m.predict_proba(X) for m in self.models], axis=0)
        return probs.mean(axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
