from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(xtrain, ytrain):
    model = LogisticRegression()
    model.fit(xtrain, ytrain)
    return model

def train_random_forest(xtrain, ytrain):
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features='sqrt')  
    model.fit(xtrain, ytrain)
    return model