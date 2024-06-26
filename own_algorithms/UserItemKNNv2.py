from surprise import AlgoBase, Prediction, Dataset
from surprise import PredictionImpossible
from surprise import KNNBasic
from surprise.model_selection import train_test_split


class UserItemKNNv2(AlgoBase):
    def __init__(self, i_weight=0.5, u_weight=0.5):
        AlgoBase.__init__(self)

        # create sim option dictionarys
        self.sim_options_user = {'name': 'Pearson', 'user_based': True}
        self.sim_options_item = {'name': 'cosine', 'user_based': False}
        # set weight variables
        self.i_weight = i_weight
        self.u_weight = u_weight
        # create reccomendation components
        self.user_algo = KNNBasic(k=55, sim_options={'name': 'Pearson', 'user_based': True}, verbose=False)
        self.item_algo = KNNBasic(k=60, sim_options={'name': 'cosine', 'user_based': False}, verbose=False)
        self.user_algo = KNNBasic(k=55, sim_options=self.sim_options_user, verbose=False)
        self.item_algo = KNNBasic(k=60, sim_options=self.sim_options_item , verbose=False)

    def fit(self, trainset):
        # train both components
        AlgoBase.fit(self, trainset)
        self.user_algo.fit(trainset)
        self.item_algo.fit(trainset)

    def estimate(self, u, i):
        # generate predictions for each model and combine
        user_prediction = self.user_algo.predict(u, i).est
        item_prediction = self.item_algo.predict(u, i).est
        return self.u_weight * user_prediction + (1 - self.i_weight) * item_prediction


    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        # return prediction object if possible
        try:
            est = self.estimate(uid, iid)
            return Prediction(uid, iid, r_ui, est, details=None)
        except PredictionImpossible as e:
            if verbose:
                print(str(e))
            return None



