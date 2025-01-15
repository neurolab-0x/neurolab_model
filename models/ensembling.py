from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_stacking(x_train, y_train):
  base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(user_label_encoder=False, eval_metric='mlogloss' ,random_state=42))
  ]
  meta_learner = LogisticRegression()
  model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)
  model.fit(x_train, y_train)
  return model

def train_voting(x_train, y_train):
  base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss' ,random_state=42))
  ]
  model = VotingClassifier(estimators=base_learners, voting='soft')
  model.fit(x_train, y_train)

  return model