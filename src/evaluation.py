from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(model, xtrain, ytrain, xtest, ytest):
    kfold = KFold(n_splits=5)
    scores = cross_val_score(model, xtrain, ytrain, cv=kfold)
    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())
    
    ypred = model.predict(xtest)
    print("Test accuracy:", accuracy_score(ytest, ypred))
    print("Confusion matrix:\n", confusion_matrix(ytest, ypred))