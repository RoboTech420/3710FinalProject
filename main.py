# MODULES
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

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

# VARIABLES
trainingSize = 0.75
dataSize = 1.00 - trainingSize #just need to adjust line above

# load data
df = pd.read_csv('DataSets\Diabetes\diabetic_data.csv')

# create DataFrame
df = pd.DataFrame(df)

# extract lower and upper bounds using regular expression
df['age_lower_bound'] = df['age'].str.strip('[]').str.split('-', expand=True)[0].astype(int)
df['age_upper_bound'] = df['age'].str.strip('[]').str.split('-', expand=True)[1].str.rstrip(')').astype(int)

# drop columns that are not relevant to the analysis
df.drop(['diag_1','diag_2','diag_3','weight', 'age', 'medical_specialty', 'payer_code', 'encounter_id','patient_nbr', 'admission_type_id', 'discharge_disposition_id','admission_source_id'], axis=1, inplace=True)

# replace '?' with 'Other' in the 'profession' column
df['race'] = df['race'].replace('?', 'Other')


print(df)

# print the columns
for coloumn in df.columns:
    print(coloumn)
    print(df[coloumn].value_counts())

# get the count of the number of columns
num_columns = df.shape[1]
print("Num coloumns: "+str(num_columns))

print(df['race'].value_counts())

for coloumn in df.columns:
    print(coloumn)


# Check column contains Particular value of DataFrame by Pandas.Series.isin()
'''
newData = df['readmitted'].isin(['NO']).head(20)
print(newData)
'''

print(df.shape)

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
