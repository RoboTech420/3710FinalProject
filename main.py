# MODULES
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns


# VARIABLES
testSize = 0.25

# FUNCTIONS
# just a random print function for now
def plotLinear(x, xLab, y, yLab):
	plt.plot(x, y)
	plt.xlabel(xLab)
	plt.ylabel(yLab)
	plt.show() 
	return

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

# load data
df = pd.read_csv('DataSets\Diabetes\diabetic_data.csv')

# create DataFrame
df = pd.DataFrame(df)

# drop columns that are not relevant to the analysis
df.drop(['diag_1','diag_2','diag_3','weight', 'medical_specialty', 'payer_code', 'encounter_id','patient_nbr', 'admission_type_id', 'discharge_disposition_id','admission_source_id'], axis=1, inplace=True)

# replace '?' with 'Other' in the 'race' column
df['race'] = df['race'].replace('?', 'Other')

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

# Decode the encoded values in each column and print the DataFrame with the decoded values
for column, encoder in label_encoders.items():
    df[column] = encoder.inverse_transform(df[column])
    


# Displays the heat map of the data
sns.heatmap(df.corr(numeric_only=True))
plt.show()

# Race Graph vs Count of Patients
plt.hist(df['race'])
plt.title('Patient Race Distribution')
plt.xlabel('Race')
plt.ylabel('Count')
plt.show()

print(df)


#sns.lineplot(df)
#plt.show()

#for column in df.columns:
	#decisionTree(df, column)
	

