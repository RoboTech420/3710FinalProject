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

# Plot Race vs Readmitted
def plot_race_countplot(df):
    sns.countplot(data=df, x='race', hue='readmitted')
    plt.title('Race vs Readmitted')
    plt.legend(loc='upper right')
    plt.show()

# Plot Age vs Readmitted
def plot_age_countplot(df):
    sns.countplot(data=df, x='age', hue='readmitted')
    plt.title('Age vs Readmitted')
    plt.legend(loc='upper right')
    plt.show()

# Plot Gender vs Readmitted
def plot_gender_countplot(df):
    sns.countplot(data=df, x='gender', hue='readmitted')
    plt.title('Gender vs Readmitted')
    plt.legend(loc='upper right')
    plt.show()

# Plot Diabetes vs Readmitted
def plot_diabetes_countplot(df):
    sns.countplot(data=df, x='diabetesMed', hue='readmitted')
    plt.title('Diabetes vs Readmitted')
    plt.legend(loc='upper right')
    plt.show()

# Plot Time in Hospital VS. Readmission
def plot_hospital_countplot(df):
    fig = plt.figure(figsize=(12,6))
    ax=sns.kdeplot(df.loc[(df['readmitted'] == "NO"),'time_in_hospital'] , color='b',fill=True,label='No readmittion')
    ax=sns.kdeplot(df.loc[(df['readmitted'] == ">30"),'time_in_hospital'] , color='r',fill=True, label='Readmittion in >30 days')
    ax=sns.kdeplot(df.loc[(df['readmitted'] == "<30"),'time_in_hospital'] , color='y',fill=True, label='Readmitted in <30 days')
    ax.set(xlabel='Time in Hospital (days)', ylabel='Frequency')
    plt.title('Time in Hospital VS. Readmitted')
    plt.legend(loc='upper right')
    plt.show()

# Bar plot to compare the F1 score and accuracy
def plot_compare_f1_acc(results):
    # Create a bar plot
    fig, ax = plt.subplots()
    models = list(results.keys())
    accuracy = [val['accuracy'] for val in results.values()]
    f1_score = [val['f1_score'] for val in results.values()]

    bar_width = 0.35
    x_pos = [i for i, _ in enumerate(models)]

    plt.bar(x_pos, accuracy, color='b', width=bar_width, label='Accuracy')
    plt.bar([i + bar_width for i in x_pos], f1_score, color='g', width=bar_width, label='F1 Score')

    plt.xlabel("Models")
    plt.ylabel("Scores")
    plt.title("Comparison of F1 Score and Accuracy for Each Model")
    plt.xticks([i + bar_width/2 for i in x_pos], models)
    plt.legend()
    plt.show()

# Pie Chart of the Readmitted Data
def plot_pie_chart(df):
    plt.pie(df['readmitted'].value_counts(), labels=df['readmitted'])
    plt.title("Comparison of Readmitted Data")
    plt.show()