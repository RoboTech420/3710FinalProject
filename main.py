# MODULES
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB # 1. choose model class
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

# 
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
    
print(df)


#for column in df.columns:
	#decisionTree(df, column)

X_set = df[['race','num_procedures','num_lab_procedures']]
Y_set = df['diabetesMed']

Xtrain, Xtest, ytrain, ytest = train_test_split(X_set, Y_set, random_state=1)
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data
print(accuracy_score(ytest, y_model))


#DECODES DATASET
# Decode the encoded values in each column and print the DataFrame with the decoded values
for column, encoder in label_encoders.items():
    df[column] = encoder.inverse_transform(df[column])
    
print(df)