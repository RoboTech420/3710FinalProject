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
from sklearn.metrics import accuracy_score
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

# Decision Tree
# Implements a decision tree classifier on a given dataset and target column. It then splits the dataset into training and
# testing sets, fits the classifier on the training data, and evaluates its performance on the testing data using the accuracy
# score. Finally, it prints the accuracy score and the target column for reference.
def decisionTree(df, col):
	X = df.drop(col, axis=1)
	y = df[col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
	clf = DecisionTreeClassifier(random_state=42)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f'Accuracy : {accuracy} {col}')
	return

#MAIN

df = ReadDataset.load_data()

#ENCODES DATASET
# Create a LabelEncoder object for each column and fit them on the respective column
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        label_encoders[column].fit(df[column])
# Encode the values in each column and print the DataFrame with the encoded values
for column, encoder in label_encoders.items():
    df[column] = encoder.transform(df[column])
# prints the dataframe    
print(df)



X_set = df[['race','num_procedures','num_lab_procedures']]
Y_set = df['diabetesMed']

print(Y_set.value_counts())
OurGraphs.racemap(df)
# GAUSSIAN
# Use the Gaussian Naive Bayes algorithm to classify data into two classes (i.e., binary classification).
# It splits the dataset into training and testing sets using train_test_split function from scikit-learn,
# then fits a GaussianNB model on the training data, predicts the class labels of the testing data using the predict method,
# and finally calculates the accuracy score of the predicted labels using the accuracy_score function from scikit-learn.
Xtrain, Xtest, ytrain, ytest = train_test_split(X_set, Y_set, random_state=1)
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data
print(accuracy_score(ytest, y_model))

# DECISION TREE

for column in df.columns:
	decisionTree(df, column)



# CROSS-VALIDATION
# We are using logistic regression as the model to evaluate, but you can use any other model as per your requirements.
# The cross_val_score function will return an array of accuracy scores for each fold. You can then compute the mean and
# standard deviation of the scores to get an estimate of the model's accuracy on unseen data. 
# Define the number of folds
k = 10
# Create a cross-validation object using KFold
cv = KFold(n_splits=k, shuffle=True, random_state=42)
# Create the model you want to evaluate
model = LogisticRegression()
# Evaluate the model using cross_val_score
scores = cross_val_score(model, X_set, Y_set, cv=cv)
# Print the scores
print("Scores: ", scores)
print("Mean Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())


#DECODES DATASET
# Decode the encoded values in each column and print the DataFrame with the decoded values
for column, encoder in label_encoders.items():
    df[column] = encoder.inverse_transform(df[column])
# prints the dataframe    
print(df)