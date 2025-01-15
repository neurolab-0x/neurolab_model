from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest_with_grid_search(x_train, y_train):
  param_grid = {
    'n_estimators' : [100, 200, 300],
    'max_depth' : [10, 20, None],
    'min_samples_split' : [2, 5, 10],
  }
  grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', verbose=1)
  grid_search.fit(x_train, y_train)
  return grid_search.best_estimator_, grid_search.best_params_