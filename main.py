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

#Reading data from the URL
'''
url="https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
urlData=pd.read_csv(url, header=None)
#print(urlData)
#parseData(urlData)

'''

#Reading data names STILL NEEDS TO BE PARSED
'''
url="https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names"
datasetName=pd.read_csv(url)
print(datasetName)
'''

#Read dataset from file
dataFrame = pd.read_csv("DataSets\Diabetes\diabetic_data.csv")
print(dataFrame.head(20))
trainingData = len(dataFrame) * trainingSize

# Check column contains Particular value of DataFrame by Pandas.Series.isin()
newData = dataFrame['readmitted'].isin(['NO']).head(20)
print(newData)


#Plotting ALL data, re-opened the csv
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["race", "readmitted"]
df = pd.read_csv("DataSets\Diabetes\diabetic_data.csv", usecols=columns)
print("Contents in csv file:", df)
plt.plot(df.race, df.readmitted)
plt.show()

# This prints only the female genders and cauacsian
female_df = dataFrame[(dataFrame['gender'] == 'Female') & (dataFrame['race'] == 'Caucasian')]
print(female_df)



#Reads Two Columns into x, y and merges them together 
'''
x = dataFrame["encounter_id"]
y = dataFrame["time_in_hospital"]
df = pd.merge(x,y, right_index=True, left_index=True)
print(df)
'''
