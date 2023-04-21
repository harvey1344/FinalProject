import pandas as pd

from own_algorithms.helper import get_top_n


def get_top_n_list(predictions, n, user, movies):
    top_n = get_top_n(predictions, n)
    user_list = top_n.get(user)
    raw_item_ids = [t[0] for t in user_list]
    int_list = [int(s) for s in raw_item_ids]
    movie_names = movies.loc[movies['movie_id'].isin(int_list), 'title'].tolist()
    return movie_names
