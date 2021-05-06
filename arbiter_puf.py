import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class Arbiter_PUF(object):
    def __init__(self, n_stages, n_crp, challenge_vector=None):
        self.n_stages = n_stages
        self.n_crp = n_crp
        self.stage_delays = np.random.normal(size=n_stages+1)

        self.challenge_vector = challenge_vector


        self.x = []
        self.y = []



    def generate_response(self):
        challenge_vector = np.random.randint(2, size=self.n_stages) if self.challenge_vector is None\
            else self.challenge_vector

        feature_vector = []
        for i in range(self.n_stages):
            feature = 1
            for j in range(i, self.n_stages):
                feature = feature * pow(-1, challenge_vector[j])
            feature_vector.append(feature)
        feature_vector.append(1)

        puf_result = np.dot(self.stage_delays, feature_vector) > 0
        return feature_vector, puf_result

    def generate_dataset(self):
        for _ in range(self.n_crp):
            feature_vector, puf_result = self.generate_response()

            self.x.append(feature_vector)
            self.y.append(puf_result)

        self.x = np.array(self.x)
        self.y = np.array(self.y)

if __name__ == '__main__':
    n_stages = [32, 64, 128, 256, 512]
    clf_score = 0
    clf_crp = 0

    svc_score = 0
    svc_crp = 0

    for n_stage in n_stages:
        n_crp = 100
        flag = True
        clf_score = 0
        svc_score = 0


        while flag:
            arbiter = Arbiter_PUF(n_stages=n_stage, n_crp=n_crp)
            arbiter.generate_dataset()
            X_train, X_test, Y_train, Y_test = train_test_split(arbiter.x, arbiter.y, test_size=0.2, random_state=0)

            if clf_score < 0.90:
                clf = LogisticRegression(solver="lbfgs").fit(X_train, Y_train)
                Y_pred = clf.predict(X_test)
                clf_score = clf.score(X_test, Y_test)
                clf_crp = n_crp


            if svc_score < 0.90:
                svc = SVC(kernel="linear").fit(X_train, Y_train)
                Y_pred = svc.predict(X_test)
                svc_score = svc.score(X_test, Y_test)
                svc_crp = n_crp

            n_crp += 100
            # print(n_crp)
            flag = ((clf_score < 0.90) or (svc_score < 0.90)) and (n_crp< 5000)


        print("n_stages:", n_stage)
        print("linear regression accuracy", clf_score)
        print("crps needed", clf_crp)
        print("============================")
        print("svm regression accuracy", svc_score)
        print("crps needed", svc_crp)
        print("\n")










