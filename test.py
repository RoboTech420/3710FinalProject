# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import re

# load data
df = pd.read_csv('DataSets\Diabetes\diabetic_data.csv')

# create DataFrame
df = pd.DataFrame(df['age'])

# define regular expression pattern for age range lower and upper bound
pattern = r'\[(\d+)\-(\d+)\)'

# extract lower and upper bounds using regular expression
df['lower_bound'] = df['age'].str.strip('[]').str.split('-', expand=True)[0].astype(int)
df['upper_bound'] = df['age'].str.strip('[]').str.split('-', expand=True)[1].str.rstrip(')').astype(int)

# extract lower and upper bounds using regular expression
match = re.search(pattern, df['age'][0])
lower_bound = int(match.group(1))
upper_bound = int(match.group(2))

# print results
print(df)
print('Lower Bound:', lower_bound)
print('Upper Bound:', upper_bound)

# drop columns that are not relevant to the analysis
df.drop(['age'], axis=1, inplace=True)

print(df)

'''
# replace '?' with NaN
df = df.replace('?', np.nan)

# Create new columns for lower and upper bounds
df['lower_bound'] = df['age'].str.split('-', expand=True)[0].astype(int)
df['upper_bound'] = df['age'].str.split('-', expand=True)[1].str.replace(')', '').astype(int)

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