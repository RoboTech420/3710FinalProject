# MODULES
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB # 1. choose model class
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import ReadDataset
import OurGraphs


# VARIABLES
testSize = 0.25

# FUNCTIONS
# prints the columns
def printColumns(df):
     # print the columns
	for coloumn in df.columns:
		print(coloumn)
		print(df[coloumn].value_counts())
	return

# MAIN

# LOAD THE DATA
df = ReadDataset.load_data()

# ENCODES DATASET
# Create a LabelEncoder object for each column and fit them on the respective column
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        label_encoders[column].fit(df[column])
# Encode the values in each column and print the DataFrame with the encoded values
for column, encoder in label_encoders.items():
    df[column] = encoder.transform(df[column])


# Select the coloumn we want to train for
X_set = df.drop(['readmitted'], axis=1)
Y_set = df['readmitted']


# Prints the value counts
print(Y_set.value_counts())

# GAUSSIAN
# Use the Gaussian Naive Bayes algorithm to classify data into two classes (i.e., binary classification).
# It splits the dataset into training and testing sets using train_test_split function from scikit-learn,
# then fits a GaussianNB model on the training data, predicts the class labels of the testing data using the predict method,
# and finally calculates the accuracy score of the predicted labels using the accuracy_score function from scikit-learn.
Xtrain, Xtest, ytrain, ytest = train_test_split(X_set, Y_set, random_state=42)
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data
print(accuracy_score(ytest, y_model))

# DECISION TREE
# Implements a decision tree classifier on a given dataset and target column. It then splits the dataset into training and
# testing sets, fits the classifier on the training data, and evaluates its performance on the testing data using the accuracy
# score. Finally, it prints the accuracy score and the target column for reference.
clf = DecisionTreeClassifier(max_depth=10, random_state=42)
clf.fit(Xtrain, ytrain)
y_pred = clf.predict(Xtest)
accuracy = accuracy_score(ytest, y_pred)
print(f'Decision Tree Accuracy : {accuracy}')


# CROSS-VALIDATION
# We are using logistic regression as the model to evaluate, but you can use any other model as per your requirements.
# The cross_val_score function will return an array of accuracy scores for each fold. You can then compute the mean and
# standard deviation of the scores to get an estimate of the model's accuracy on unseen data. 
# Define the number of folds
k = 10
# Create a cross-validation object using KFold
cv = KFold(n_splits=k, shuffle=True, random_state=42)
# Create the model you want to evaluate
model = LogisticRegression(max_iter=1000)
# Evaluate the model using cross_val_score
scores = cross_val_score(model, X_set, Y_set, cv=cv)
# Print the scores
print("Scores: ", scores)
print("Mean Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())


# HYPERPARAMETER TUNING
'''
print("Hyperparameter Tuning: ")
params = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'class_weight': [None, 'balanced'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 500, 1000, 5000]
}
cv = KFold(n_splits=k, shuffle=True, random_state=42)
model = LogisticRegression()
random_search = RandomizedSearchCV(model, param_distributions=params, cv=cv, n_iter=100, n_jobs=-1, random_state=42)
random_search.fit(X_set, Y_set)
print('Best parameters:', random_search.best_params_)
print('Best score:', random_search.best_score_)
'''



#DECODES DATASET
# Decode the encoded values in each column and print the DataFrame with the decoded values
for column, encoder in label_encoders.items():
    df[column] = encoder.inverse_transform(df[column])