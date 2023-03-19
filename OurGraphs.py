import matplotlib.pyplot as plt
import seaborn as sns

# just a random print function for now
def plotLinear(x, xLab, y, yLab):
	plt.plot(x, y)
	plt.xlabel(xLab)
	plt.ylabel(yLab)
	plt.show() 
	return

# Displays the heat map of the data
def heatmap(df):
    sns.heatmap(df.corr(numeric_only=True))
    plt.show()
    return

# Race Graph vs Count of Patients
def racemap(df):
    plt.hist(df['race'])
    plt.title('Patient Race Distribution')
    plt.xlabel('Race')
    plt.ylabel('Count')
    plt.show()
    return

# SNL Plot 
def snlPlot(df):
    sns.lineplot(df)
    plt.show()
    return