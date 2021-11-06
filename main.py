import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import algorithms

''' ADD transactions.csv UNDER THE DATA FOLDER (it cant be pushed due to the size so never push it) '''
transactions = pd.read_csv(r'Data\transactions.csv')

# Reordering the columns so is_fraud is on the very right
cols = transactions.columns.tolist()
cols = cols[:-3] + cols[-2:]
transactions2 = transactions[cols]
transactions2 = transactions2.assign(is_fraud=transactions.loc[:, "is_fraud"])

# --------------- PREPROCESSING ---------------
# X -> features, y -> label
X = transactions.drop(['is_fraud'], axis=1)
y = transactions['is_fraud']

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=10)

# ----- PREPROCESS TRAINING DATA -----

t_source = X_train.loc[:, 't_source']
t_destination = X_train.loc[:, 't_destination']
t_type = X_train.loc[:, 't_type']

leSource = preprocessing.LabelEncoder().fit(t_source)
leDest = preprocessing.LabelEncoder().fit(t_destination)
leType = preprocessing.LabelEncoder().fit(t_type)

letranssource = leSource.transform(t_source)
letransdestination = leDest.transform(t_destination)
letranstype = leType.transform(t_type)

X_train.loc[:, 't_source'] = letranssource
X_train.loc[:, 't_destination'] = letransdestination
X_train.loc[:, 't_type'] = letranstype

# ----- PREPROCESS TESTING DATA -----

t_source_test = X_test.loc[:, 't_source']
t_destination_test = X_test.loc[:, 't_destination']
t_type_test = X_test.loc[:, 't_type']

leSource_test = preprocessing.LabelEncoder().fit(t_source_test)
leDest_test = preprocessing.LabelEncoder().fit(t_destination_test)
leType_test = preprocessing.LabelEncoder().fit(t_type_test)

letranssource_test = leSource_test.transform(t_source_test)
letransdestination_test = leDest_test.transform(t_destination_test)
letranstype_test = leType_test.transform(t_type_test)

X_test.loc[:, 't_source'] = letranssource_test
X_test.loc[:, 't_destination'] = letransdestination_test
X_test.loc[:, 't_type'] = letranstype_test

# Replace NaNs with 0's
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# ------ Print accuracies ------

# print("Decision Tree accuracy:")
# dtree = algorithms.dtree(X_train, X_test, y_train, y_test)

# print("kNN accuracy:")
# kNN = algorithms.kNN(X_train, X_test, y_train, y_test)

print("Naive Bayes results:")
naiveB = algorithms.naiveB(X_train, X_test, y_train, y_test)

# print("Random Forest accuracy:")
# rforest = algorithms.rforest(X_train, X_test, y_train, y_test)



