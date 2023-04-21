import pandas as pd
import numpy as np
from surprise import Dataset, NMF
from surprise.model_selection import GridSearchCV

data = Dataset.load_builtin('ml-100k')
algo = NMF()
print("Hello")

param_grid = {'n_factors':[20,50,100,200],
              'lr_bu':[0.005, 0.01, 0.02],
              'lr_bi':[0.005, 0.01, 0.02],
              'reg_bu':[0.05,0.1,0.2],
              'reg_bi':[0.05,0.1,0.2],
              'n_epochs':[40,50,60,70],
              'biased':[False, True]}

grid_search = GridSearchCV(algo_class=NMF, param_grid=param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
grid_search.fit(data)

# Print the best parameters and score
print("Best RMSE score:", grid_search.best_score['rmse'])
print("Best MAE score:", grid_search.best_score['mae'])
print("Best parameters:", grid_search.best_params['rmse'])

nmf_params = grid_search.best_estimator['rmse']
df = pd.DataFrame(data=[grid_search.best_params['rmse']])
df.to_csv('./NMF_rmse_best.csv', index=False)