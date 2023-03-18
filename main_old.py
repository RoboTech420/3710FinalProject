# MODULES
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# FUNCTIONS
def parseData(urlData):
	print(urlData)
	return

def plotLinear(x, xLab, y, yLab):
	plt.plot(x, y)
	plt.xlabel(xLab)
	plt.ylabel(yLab)
	plt.show() 
	return

def printColumns(df):
     # print the columns
	for coloumn in df.columns:
		print(coloumn)
		print(df[coloumn].value_counts())
	return

# VARIABLES
trainingSize = 0.75
dataSize = 1.00 - trainingSize #just need to adjust line above

# load data
df = pd.read_csv('DataSets\Diabetes\diabetic_data.csv')

# create DataFrame
df = pd.DataFrame(df)

# drop columns that are not relevant to the analysis
df.drop(['diag_1','diag_2','diag_3','weight', 'medical_specialty', 'payer_code', 'encounter_id','patient_nbr', 'admission_type_id', 'discharge_disposition_id','admission_source_id'], axis=1, inplace=True)

# replace '?' with 'Other' in the 'race' column
df['race'] = df['race'].replace('?', 'Other')

# Create an instance of the LabelEncoder
label_encoder = LabelEncoder()

# Iterate over each categorical variable and encode its values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

print(df)

# get the count of the number of columns
num_columns = df.shape[1]
print("Num coloumns: "+str(num_columns))

print(df['race'].value_counts())

for coloumn in df.columns:
    print(coloumn)


print(df['num_procedures'].value_counts())
print(df['insulin'].value_counts())

# Check column contains Particular value of DataFrame by Pandas.Series.isin()
'''
newData = df['readmitted'].isin(['NO']).head(20)
print(newData)
'''

print(df.shape)
print(df.dtypes)
print(df.info())

# Displays the heat map of the data
sns.heatmap(df.corr(numeric_only=True))
plt.show()

# Instantiate the LabelEncoder
le = LabelEncoder()

# Fit the LabelEncoder to the data
le.fit(df['race'])

# Transform the data to numerical form
data_encoded = le.transform(df['race'])
print(data_encoded)

# Decode the numerical data back to its original form
data_decoded = le.inverse_transform(data_encoded)
print(data_decoded)
print(le)

#Plotting ALL data, re-opened the csv
'''
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["race", "readmitted"]
df = pd.read_csv("DataSets\Diabetes\diabetic_data.csv", usecols=columns)
print("Contents in csv file:", df)
plt.plot(df.race, df.readmitted)
plt.show()
'''

#Reads Two Columns into x, y and merges them together 
'''
x = dataFrame["encounter_id"]
y = dataFrame["time_in_hospital"]
df = pd.merge(x,y, right_index=True, left_index=True)
print(df)
'''
