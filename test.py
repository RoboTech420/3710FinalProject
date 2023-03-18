# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# load data
df = pd.read_csv('DataSets\Diabetes\diabetic_data.csv')

# create DataFrame
df = pd.DataFrame(df)

# drop columns that are not relevant to the analysis
df.drop(['diag_1','diag_2','diag_3','weight', 'age', 'medical_specialty', 'payer_code', 'encounter_id','patient_nbr', 'admission_type_id', 'discharge_disposition_id','admission_source_id'], axis=1, inplace=True)

# replace '?' with 'Other' in the 'profession' column
df['race'] = df['race'].replace('?', 'Other')


# print the columns
for column in df.columns:
    print(column)
    print(df[column].value_counts())

# get the count of the number of columns
num_columns = df.shape[1]
print("Num coloumns: "+str(num_columns))

print(df['race'].value_counts())

for column in df.columns:
    print(column)


# Create a LabelEncoder object for each column and fit them on the respective column
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        label_encoders[column].fit(df[column])

# Encode the values in each column and print the DataFrame with the encoded values
for column, encoder in label_encoders.items():
    df[column] = encoder.transform(df[column])
    
# Decode the encoded values in each column and print the DataFrame with the decoded values
for column, encoder in label_encoders.items():
    df[column] = encoder.inverse_transform(df[column])
    
print("\nDecoded DataFrame:")
print(df)
print(len(label_encoders))

print("Encoded DataFrame:")
print(df)
plt.hist(df['race'])
plt.title('Patient Race Distribution')
plt.xlabel('Race')
plt.ylabel('Count')
plt.show()

plt.scatter(df['time_in_hospital'], df['num_medications'])
plt.title('Time in Hospital vs Number of Medications')
plt.xlabel('Time in Hospital')
plt.ylabel('Number of Medications')
plt.show()

'''
# Decode the encoded values in each column and print the DataFrame with the decoded values
for column, encoder in label_encoders.items():
    if df[column].dtype == 'object':
        df[column + '_Decoded'] = encoder.inverse_transform(df[column + '_Encoded'])
print("\nDecoded DataFrame:")
print(df)
'''


# possible lable encoder
'''
col = df['race']
# Instantiate the LabelEncoder
le = LabelEncoder()

# Fit the LabelEncoder to the data
le.fit(col)

# Transform the data to numerical form
data_encoded = le.transform(col)
print(data_encoded)

# Decode the numerical data back to its original form
data_decoded = le.inverse_transform(data_encoded)
print(data_decoded)
'''

'''
# replace '?' with NaN
df = df.replace('?', np.nan)

# drop rows with missing or invalid values
df.dropna(inplace=True)
df = df[df['gender'] != 'Unknown/Invalid']

# convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['race', 'gender', 'age'])


# drop columns that are not relevant to the analysis
#df.drop(['admission_type_id', 'discharge_disposition_id', 'admission_source_id'], axis=1, inplace=True)
'''

'''
# split data into training and testing sets
X = df.drop(['readmitted'], axis=1)
y = df['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluate model performance
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
'''

