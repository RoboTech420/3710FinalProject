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

#Read dataset from file
dataFrame = pd.read_csv("DataSets\\test.csv")
print(dataFrame)
#plotLinear(dataFrame["area"],"area",dataFrame["price"],"price")


#Linear Regression Single Variable
reg = linear_model.LinearRegression()
reg.fit(dataFrame[["area"]].values,dataFrame["price"].values)
print(reg.predict([[3300]])) #predicts price for 3300 sq ft
#print(reg.coef_)
#print(reg.intercept_)
#y=mb+b
print(reg.coef_ * 3300 + reg.intercept_)

d = pd.read_csv("DataSets\\area.csv")
print(d.head(3))
p = reg.predict(d)
d['prices'] = p
d.to_csv("prediction.csv", index=False)

plt.xlabel("area", fontsize=20)
plt.ylabel("price", fontsize=20)
plt.scatter(dataFrame["area"],dataFrame["price"], color="red", marker="+")
plt.plot(dataFrame["area"], reg.predict(dataFrame[["area"]]), color="blue")
plt.show()