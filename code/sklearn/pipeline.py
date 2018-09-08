# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import sklearn as sk

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.model_selection import learning_curve

from sklearn.neural_network import MLPClassifier

from sklearn.externals import joblib

from preparator import prepare_data, split_x_y_group, drop_N_groups


df_train = pd.read_csv('train_sample.csv')
df_test = pd.read_csv('test_sample.csv')


df = df_train.copy()
df = df.append(df_test)
        
data = prepare_data(df)

# data1, count = drop_N_groups(data, 750)
# print count


X, y, grp = split_x_y_group(data)


def doAll(X, y, grp, scoring='accuracy', model_file="model.pkl",
          validation_file='validation.csv', train_score_file="train_score_csv", train_sizes_file="train_sizes.csv"):
       
    pipe_net = Pipeline([
            ('clf', MLPClassifier(max_iter=300, hidden_layer_sizes=(5, 2), alpha=0.01, random_state=1))
    ])
    
    
    gkf = GroupKFold(n_splits=10)

    
    param_range = [(3, 2), (5, 3), (9, 5), (15, 7), (20, 15), (25, 17)]
    param_grid = [{'clf__hidden_layer_sizes': param_range}]
    gs = GridSearchCV(
        estimator=pipe_net,
        param_grid=param_grid,
        scoring=scoring,
        cv=gkf,
        n_jobs=4
    )
        
    gs = gs.fit(X, y, groups=grp)
    best = gs.best_estimator_

    train_sizes, train_scores, valid_scores = learning_curve(
        best, X, y, train_sizes=np.linspace(0.1, 1.0, 10), 
        cv=gkf, groups=grp, scoring=scoring, n_jobs=4)
    
    np.savetxt(validation_file, valid_scores, delimiter=',')
    np.savetxt(train_sizes_file, train_sizes, delimiter=',')
    np.savetxt(train_score_file, train_scores, delimiter=',')
    
    joblib.dump(best, model_file) 
    
    


doAll(X, y, grp, 'accuracy')