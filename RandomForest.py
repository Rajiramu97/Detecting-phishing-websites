# -*- coding: utf-8 -*-


#----------------importing libraries
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

warnings.filterwarnings('ignore')
#importing the dataset
dataset = pd.read_csv("datasets/phishcoop.csv")
dataset = dataset.drop('id', 1) #removing unwanted column

x = dataset.iloc[ : , :-1].values
y = dataset.iloc[:, -1:].values

#spliting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.40, random_state =0)

#----------------applying grid search to find best performing parameters 
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [100, 700],
    'max_features': ['sqrt', 'log2'],
    'criterion' :['gini', 'entropy']}]

"""grid_search = GridSearchCV(RandomForestClassifier(),  parameters,cv =5, n_jobs= -1)
grid_search.fit(x_train, y_train)
grid_search1 = GridSearchCV(RandomForestClassifier(),  parameters,cv =5, n_jobs= -1,scoring='precision')
grid_search1.fit(x_train, y_train)
grid_search2 = GridSearchCV(RandomForestClassifier(),  parameters,cv =5, n_jobs= -1,scoring='recall')
grid_search2.fit(x_train, y_train)
grid_search3 = GridSearchCV(RandomForestClassifier(),  parameters,cv =5, n_jobs= -1,scoring='f1')
grid_search3.fit(x_train, y_train
#printing best parameters 
print("\nBest Accuracy =" +str( grid_search.best_score_))
print("\n")
print("Best Precision =" +str( grid_search1.best_score_))
print("\n")
print("Best Recall =" +str( grid_search2.best_score_))
print("\n")
print("Best Fmeasure =" +str( grid_search3.best_score_))
print("\n")
print("best parameters =" + str(grid_search3.best_params_))
#-------------------------------------------------------------------------"""

#fitting RandomForest regression with best params 
classifier = RandomForestClassifier(n_estimators=30, criterion='gini', max_depth=1, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=0.2, max_leaf_nodes=2, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=0, verbose=0, warm_start=False, class_weight=None)
classifier.fit(x_train, y_train)
classifier.fit(x_train, y_train)

#predicting the tests set result
y_pred = classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix\n")
print(cm)
tp=cm[0][0]
fp=cm[0][1]
fn=cm[1][0]
tn=cm[1][1]
print("\nAccuracy\n")
acc=(tp+tn)/(tp+tn+fp+fn)
print(acc)
recall=(tp)/(tp+fn)
print("\nRecall\n")
print(recall)
pre=(tp)/(tp+fp)
print("\nPrecision\n")
print(pre)
fmeasure=(2*recall*pre)/(recall+pre)
print("\nFmeasure\n")
print(fmeasure)
print("\nspecificity\n")
specificity=tn/(tn+fp)
print(specificity)
print("\nfalse postive rate\n")
fpr=fp/(fp+tn)
print(fpr)

#pickle file joblib
joblib.dump(classifier, 'final_models/rf_final.pkl')



