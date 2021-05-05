from functools import reduce

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from arbiter_puf import Arbiter_PUF


class XOR_arbiter_PUF(object):
    def __init__(self, n_stages, n_crp, challenge_vector=None):
        self.n_stages = n_stages
        self.n_crp = n_crp
        self.challenge_vector = challenge_vector

        self.x = []
        self.y = []

    def generate_response(self):
        feature_vectors = []
        puf_results = []


        for stage in self.n_stages:
            challenge_vector = np.random.randint(2, size=stage) if self.challenge_vector is None \
                else self.challenge_vector

            puf = Arbiter_PUF(n_stages=stage, n_crp=1000, challenge_vector=challenge_vector)
            feature_vector, puf_result = puf.generate_response()

            feature_vectors.append(feature_vector)
            puf_results.append(puf_result)

        response = reduce(lambda x, y: x ^ y, puf_results)

        return feature_vectors, response

    def generate_dataset(self):
        for i in range(self.n_crp):
            feature_vector, puf_result = self.generate_response()

            self.x.append(feature_vector)
            self.y.append(puf_result)

        self.x = np.array(self.x)
        self.y = np.array(self.y)


if __name__ == '__main__':
    n_stages = [32, 32, 32, 32]
    xor_puf = XOR_arbiter_PUF(n_stages=n_stages, n_crp=1000)
    xor_puf.generate_dataset()

    X_train, X_test, Y_train, Y_test = train_test_split(xor_puf.x, xor_puf.y, test_size=0.2, random_state=0)

    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples, nx*ny))
    clf = LogisticRegression(solver="lbfgs").fit(X_train, Y_train)

    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    Y_pred = clf.predict(X_test)
    clf_score = clf.score(X_test, Y_test)
    print(clf_score)
