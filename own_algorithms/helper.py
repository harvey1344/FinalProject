import time
from collections import defaultdict

import numpy as np, pandas as pd
from surprise.model_selection import KFold
from surprise.accuracy import rmse, mse, mae


def testing_algorithm(algo, data):
    # Define the K-fold cross-validation iterator with 10 splits
    kf = KFold(n_splits=5, random_state=1)

    # Initialize arrays to store the scores
    fit_times = np.array([])
    predict_times = np.array([])
    rmse_scores = np.array([])
    mse_scores = np.array([])
    mae_scores = np.array([])

    # Perform K-fold cross-validation
    for trainset, testset in kf.split(data):
        # Train the algorithm on the training set and measure the time taken
        start_fit = time.time()
        algo.fit(trainset)
        fit_times = np.append(fit_times, time.time() - start_fit)

        # Predict the ratings for the test set and measure the time taken
        start_predict = time.time()
        predictions = algo.test(testset)
        predict_times = np.append(predict_times, time.time() - start_predict)

        # Compute the RMSE, MSE, and MAE scores for the predictions
        rmse_scores = np.append(rmse_scores, rmse(predictions, verbose=False))
        mse_scores = np.append(mse_scores, mse(predictions, verbose=False))
        mae_scores = np.append(mae_scores, mae(predictions, verbose=False))

    # Return the mean of the RMSE, MSE, MAE,fit time and predict time
    return np.array(
        [np.mean(rmse_scores), np.mean(mse_scores), np.mean(mae_scores), np.mean(fit_times), np.mean(predict_times)])


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


