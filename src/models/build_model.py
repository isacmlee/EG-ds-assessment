import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle

import numpy as np

def feature_selection():
    data = pd.read_csv('data/processed/processed_sc_player_data.csv')

    features = data.drop(columns=['LeagueIndex'],axis=1)
    target = data['LeagueIndex']

    X_sel_train, X_sel_test, y_sel_train, y_sel_test = train_test_split(features, target,test_size=0.3)

    sfm = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    sfm.fit(X_sel_train, y_sel_train)
    selected_feat= features.columns[(sfm.get_support())]

    print(selected_feat)

    return selected_feat

def build_log_reg(selected_feat):
    data = pd.read_csv('data/processed/processed_sc_player_data.csv')

    features = data.drop(columns=['LeagueIndex'],axis=1)
    target = data['LeagueIndex']

    features = features[selected_feat]

    X_train, X_test, y_train, y_test = train_test_split(features, target,test_size=0.3,random_state=1)

    k_folds = KFold(n_splits = 5)

    logreg = LogisticRegression(multi_class='ovr', solver='liblinear') # one vs. rest

    log_reg_scores = cross_val_score(logreg, X_train, y_train, cv = k_folds)
    print(log_reg_scores)
    print(np.average(log_reg_scores))

    filename = 'models/log_reg_model.sav'
    pickle.dump(logreg, open(filename, 'wb'))

    return logreg

def build_nb(selected_feat):
    data = pd.read_csv('data/processed/processed_sc_player_data.csv')

    features = data.drop(columns=['LeagueIndex'],axis=1)
    target = data['LeagueIndex']

    features = features[selected_feat]

    X_train, X_test, y_train, y_test = train_test_split(features, target,test_size=0.3,random_state=1)

    gnb = GaussianNB()

    k_folds = KFold(n_splits = 5)
    nb_scores = cross_val_score(gnb, X_train, y_train, cv = k_folds)

    print(nb_scores)
    print(np.average(nb_scores))

    param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
    }

    nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
    nbModel_grid.fit(X_train, y_train)
    print(nbModel_grid.best_estimator_)

    y_preds = nbModel_grid.predict(X_test)
    print(accuracy_score(y_test, y_preds))

    filename = 'models/nb_model.sav'
    pickle.dump(nbModel_grid, open(filename, 'wb'))

    return nbModel_grid

def build_svm(selected_feat):
    data = pd.read_csv('data/processed/processed_sc_player_data.csv')

    features = data.drop(columns=['LeagueIndex'],axis=1)
    target = data['LeagueIndex']
    
    features = features[selected_feat]

    X_train, X_test, y_train, y_test = train_test_split(features, target,test_size=0.3,random_state=1)

    clf = svm.SVC()

    k_folds = KFold(n_splits = 5)

    svm_scores = cross_val_score(clf, X_train, y_train, cv = k_folds)
    print(svm_scores)
    print(np.average(svm_scores))

    filename = 'models/svm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    return clf 

if __name__ == '__main__':
    selected_features = feature_selection()
    build_log_reg(selected_features)
    build_nb(selected_features)
    build_svm(selected_features)
