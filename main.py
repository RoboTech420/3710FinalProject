# MODULES
import urllib.request
import pandas as pd
from convertData import renameData
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
renameData(urlData)
'''

#Reading data names STILL NEEDS TO BE PARSED
'''
url="https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names"
datasetName=pd.read_csv(url)
print(datasetName)
'''

#Read dataset from file
'''
dataFrame = pd.read_csv("DataSets\Diabetes\diabetic_data.csv")
print(dataFrame.head(20))
print(len(dataFrame) * traingingSize)
'''

#Reads Two Columns into x, y and merges them together 
'''
x = dataFrame["encounter_id"]
y = dataFrame["time_in_hospital"]
df = pd.merge(x,y, right_index=True, left_index=True)
print(df)
'''
