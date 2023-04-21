from surprise import AlgoBase, Prediction
from surprise import PredictionImpossible
from surprise import KNNBasic


class UserItemKNN(AlgoBase):
    def __init__(self, k=40, i_weight=0.5, u_weight=0.5):
        AlgoBase.__init__(self)

        self.k = k
        self.sim_options_user = {'name': 'cosine', 'user_based': True}
        self.sim_options_item = {'name': 'cosine', 'user_based': False}
        self.i_weight = i_weight
        self.u_weight = u_weight
        self.user_algo = KNNBasic(k=self.k, sim_options=self.sim_options_user, verbose=False)
        self.item_algo = KNNBasic(k=self.k, sim_options=self.sim_options_item, verbose=False)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.user_algo.fit(trainset)
        self.item_algo.fit(trainset)

    def estimate(self, u, i):
        user_prediction = self.user_algo.predict(u, i).est
        item_prediction = self.item_algo.predict(u, i).est
        return self.u_weight * user_prediction + (1 - self.i_weight) * item_prediction

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        try:
            est = self.estimate(uid, iid)
            return Prediction(uid, iid, r_ui, est, details=None)
        except PredictionImpossible as e:
            if verbose:
                print(str(e))
            return None

