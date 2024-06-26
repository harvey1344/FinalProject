import pandas as pd, numpy as np
from surprise import Dataset, SVDpp, SVD
from surprise.model_selection import cross_validate, GridSearchCV

data= Dataset.load_builtin('ml-100k')
algo= SVDpp()
print("Hello")

svdpp_param_grid= {'n_factors':[20,50,100,200,300,400],
             'lr_all':[0.005, 0.01, 0.02, 0.05],
             'reg_all':[0.05,0.1,0.2, 0.3],
             'n_epochs':[30,40,50,60,70]}


grid_search = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
grid_search.fit(data)

# Print the best parameters and score
print("Best RMSE score:", grid_search.best_score['rmse'])
print("Best MAE score:", grid_search.best_score['mae'])
print("Best parameters:", grid_search.best_params['rmse'])
svdpp_params=grid_search.best_estimator['rmse']
df=pd.DataFrame(data=[grid_search.best_params['rmse']])
df.to_csv('./SVDpp_rmse_best.csv', index=False)