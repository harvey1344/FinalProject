import numpy as np
from surprise import AlgoBase, Prediction, Dataset, accuracy
from surprise import PredictionImpossible
from surprise import KNNBaseline, SVD
from surprise.model_selection import train_test_split


class Combiner(AlgoBase):
    def __init__(self, knn_weight, svd_weight):
        AlgoBase.__init__(self)

        self.knn_weight = knn_weight
        self.svd_weight = svd_weight
        self.knn = KNNBaseline(k=40, verbose=False,sim_options={'name': 'pearson_baseline', 'min_support': 1, 'user_based': False,})
        self.svd = SVD(n_factors=250, lr_all=0.01, reg_all=0.1, n_epochs=50, random_state=1)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.knn.fit(trainset)
        self.svd.fit(trainset)

    def estimate(self, u, i):
        knn_prediction = self.knn.predict(u, i).est * self.knn_weight
        svd_prediction = self.svd.predict(u, i).est * self.svd_weight
        combined_prediction = knn_prediction+svd_prediction


        #print(knn_prediction, self.knn_weight, svd_prediction,self.svd_weight, combined_prediction)
        return combined_prediction


    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        try:
            est = self.estimate(uid, iid)
            return Prediction(uid, iid, r_ui, est, details=None)
        except PredictionImpossible as e:
            if verbose:
                print(str(e))
            return None

