#!/usr/bin/python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import datetime

# For terminal seperation.
print('\n')
print('-'*80)
print(' ' * 27, end='')
print(datetime.datetime.now())
print('-'*80)
print('TensorFlow Version: ', tf.VERSION)
print('Keras Version: ', tf.keras.__version__) 
print('\n')

# Data import
full_train = pd.read_csv('train.csv')
full_train.set_index('PassengerId', inplace=True)
out_test = pd.read_csv('test.csv')
out_test.set_index('PassengerId', inplace=True)
# print(full_train.head())
# print(full_train.info())
print(full_train.columns)
# print('\n')
# print(out_test.head())
# print(out_test.info())
# print(out_test.columns)

# Data processing
def process_age(df, cut_points, label_names):
	df['Age'] = df['Age'].fillna(-0.5)
	df['Age_categories'] = pd.cut(df['Age'], cut_points, labels=label_names)
	return df
cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
label_names = ['Missing', 'Infant', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
full_train = process_age(full_train, cut_points, label_names)
out_test = process_age(out_test, cut_points, label_names)
dummy_labels = []

# Data processing for others
def process_others(df, cut_points, column_name, label_names):
	df[column_name+'_categories'] = pd.cut(df[column_name], cut_points, labels=label_names)
	return df
cut_points = [-1, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 20]
label_names = ['0', '1', '2', '3', '4', '5', '6+']
full_train = process_others(full_train, cut_points, 'Parch', label_names)
full_train = process_others(full_train, cut_points, 'SibSp', label_names)
out_test = process_others(out_test, cut_points, 'Parch', label_names)
out_test = process_others(out_test, cut_points, 'SibSp', label_names)

# Data dumming
def create_dummies(df, column_name):
	dummies = pd.get_dummies(df[column_name], prefix=column_name)
	df = pd.concat([df, dummies], axis=1)
	for label in dummies.columns:
		dummy_labels.append(label)
	return df

full_train = create_dummies(full_train, 'Age_categories')
full_train = create_dummies(full_train, 'Pclass')
full_train = create_dummies(full_train, 'Sex')
full_train = create_dummies(full_train, 'SibSp_categories')
full_train = create_dummies(full_train, 'Parch_categories')

out_test = create_dummies(out_test, 'Age_categories')
out_test = create_dummies(out_test, 'Pclass')
out_test = create_dummies(out_test, 'Sex')
out_test = create_dummies(out_test, 'SibSp_categories')
out_test = create_dummies(out_test, 'Parch_categories')

# for i in full_train.columns:
# 	print(full_train.head()[i])
# 	print(full_train[i].unique())

dummy_labels = dummy_labels
	
# Split full_train to separate features and the label.
X_train_f = full_train[dummy_labels]
y_train_f = full_train['Survived']
# print(X_train_df.head())
# print(type(X_train_df))
# print(y_train_sr.head())
# print(type(y_train_sr))

# Data splitting.
X_train, X_test, y_train, y_test = train_test_split(X_train_f, y_train_f,
													test_size=0.2,
													shuffle=True)
# print('\nX_train shape', X_train.shape,
# 		'\nX_test shape', X_test.shape,
# 		'\ny_train shape', y_train.shape,
# 		'\ny_test shape', y_test.shape)
# print('X_train head\n', X_train.head())
# print('X_test head\n', X_test.head())
# print('y_train head\n', y_train.head())
# print('y_test head\n', y_test.head())

### Linear Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
scores = cross_val_score(lr, X_train_f, y_train_f, cv=10)
print(accuracy)
print(np.mean(scores))

### SVM
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
prediction = svc.predict(X_test)
accuracy = accuracy_score(prediction, y_test)
score = cross_val_score(svc, X_train_f, y_train_f, cv=10)
print(accuracy)
print(np.mean(scores))

### Predict by Linear Regression()
exam = out_test[dummy_labels]
prediction = lr.predict(exam)
exam['Survived'] = prediction
submission = exam['Survived']
print(submission)
# print(exam.head())
# print(exam.columns)
submission.to_csv('result.csv', header=['Survived'])
