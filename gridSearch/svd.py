import pandas as pd, numpy as np
from surprise import Dataset, SVD
from surprise.model_selection import cross_validate, GridSearchCV

data= Dataset.load_builtin('ml-100k')

print("Hello")

param_grid= {'n_factors':[20,50,150,200,250,300],
             'lr_all':[0.005,0.01, 0.015, 0.02, 0.05],
             'reg_all':[0.05, 0.075, 0.1, 0.15, 0.2],
             'n_epochs':[30,40,50,60,70]}


grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
grid_search.fit(data)

# Print the best parameters and score
print("Best RMSE score:", grid_search.best_score['rmse'])
print("Best MAE score:", grid_search.best_score['mae'])
print("Best parameters:", grid_search.best_params['rmse'])
svd_params=grid_search.best_estimator['rmse']
df=pd.DataFrame(data=[grid_search.best_params['rmse']])
df.to_csv('./SVD_rmse_best.csv', index=False)