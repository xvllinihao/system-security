from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from xor_arbiter import XOR_arbiter_PUF
import numpy as np

class iPUF(object):
    def __init__(self, pos, n_stage, n_puf, n_crp):
        self.n_stage = n_stage
        self.n_puf = n_puf
        self.interpose_position = pos
        self.n_crp = n_crp

        self.x = []
        self.y = []

    def generate_response(self):
        n_stages = [self.n_stage] * self.n_puf
        challenge_vector = np.random.randint(2, size=self.n_stage)
        k_up = XOR_arbiter_PUF(n_stages=n_stages, n_crp=1000, challenge_vector=challenge_vector)

        feature_vector0, puf_result = k_up.generate_response()


        new_challenge_vector = np.insert(challenge_vector,self.interpose_position,puf_result)

        n_stages = [self.n_stage+1] * self.n_puf
        k_down = XOR_arbiter_PUF(n_stages=n_stages, n_crp=1000, challenge_vector=new_challenge_vector)
        feature_vector, puf_result = k_down.generate_response()

        return feature_vector0, puf_result

    def generate_dataset(self):
        for i in range(self.n_crp):
            feature_vector, puf_result = self.generate_response()

            self.x.append(feature_vector)
            self.y.append(puf_result)

        self.x = np.array(self.x)
        self.y = np.array(self.y)

if __name__ == '__main__':
    ipuf = iPUF(pos=12, n_stage=32, n_puf=4, n_crp=5000)
    ipuf.generate_dataset()

    X_train, X_test, Y_train, Y_test = train_test_split(ipuf.x, ipuf.y, test_size=0.2, random_state=0)

    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples, nx*ny))
    clf = LogisticRegression(solver="lbfgs").fit(X_train, Y_train)

    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    Y_pred = clf.predict(X_test)
    clf_score = clf.score(X_test, Y_test)
    print(clf_score)
