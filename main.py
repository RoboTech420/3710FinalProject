# MODULES
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB # 1. choose model class
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import ReadDataset
import OurGraphs
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# VARIABLES
testSize = 0.25

# FUNCTIONS
# prints the columns
def printColumns(df):
     # print the columns
	for coloumn in df.columns:
		print(coloumn)
		print(df[coloumn].value_counts())
	return

# MAIN

# LOAD THE DATA
df = ReadDataset.load_data()

# DATA INFORMATION
print(df.info())

# Save F1 score and string name
results = {}

# ENCODES DATASET
# Create a LabelEncoder object for each column and fit them on the respective column
label_encoders = {}
for column in df.columns:
    if df[column].dtype != 'int64' and df[column].dtype != 'float64':
        df[column] = df[column].astype(str)
    label_encoders[column] = LabelEncoder()
    label_encoders[column].fit(df[column])
# Encode the values in each column and print the DataFrame with the encoded values
for column, encoder in label_encoders.items():
    df[column] = encoder.transform(df[column])

# Heat Map
#OurGraphs.heatmap(df)


# Select the coloumn we want to train for
X_set = df.drop(['readmitted'], axis=1)
Y_set = df['readmitted']
#Splits the dataset into training and testing sets using train_test_split function from scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X_set, Y_set, random_state=1)

# SCALING DATA
# Create a StandardScaler object to scale the input features
scaler = StandardScaler()
# Fit the scaler on the training set and transform both the training and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# RANDOM FOREST CLASSIFIER
# The n_estimators parameter specifies the number of trees in the forest. A larger number of trees can improve the performance of the classifier but can also increase the training time.
# The random_state parameter sets the random seed for reproducibility. By setting it to a fixed value, you ensure that the results will be the same every time you run the code.
# The score method computes the mean accuracy of the classifier on the testing data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test_scaled)
accuracy = clf.score(X_test_scaled, y_test)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Random Forest Classifier: {accuracy}")
print(f"F1 Score: {f1}")
# Save F1 score and string name
results['Random Forest F1 Score'] = {'accuracy': accuracy, 'f1_score': f1}

# GAUSSIAN NAIVE BAYES
# Define a grid of hyperparameters to search over the smoothing parameter 'var_smoothing'.
# Perform a grid search using GridSearchCV with 5-fold cross validation.
# After the grid search is complete, we print the best hyperparameters found by the search and
# use the best model to predict on the testing data.
# Print the accuracy score of the best model.
# define the hyperparameters to search over
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}
# Instantiate the model
model = GaussianNB()
# perform a grid search over the hyperparameters
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
# print the best hyperparameters
print(f"Best hyperparameters: {grid.best_params_}")
# predict on new data using the best model
best_model = grid.best_estimator_
y_model = best_model.predict(X_test_scaled)
# calculate and print the accuracy and F1 score of the best model
accuracy = accuracy_score(y_test, y_model)
f1 = f1_score(y_test, y_model, average='weighted')
print(f"Gaussian Naive Bayes Accuracy score: {accuracy}")
print(f"Gaussian Naive Bayes F1 score: {f1}")
# Save F1 score and string name
results['Gaussian Naive Bayes'] = {'accuracy': accuracy, 'f1_score': f1}


# DECISION TREE CLASSIFIER
# Implements a decision tree classifier on a given dataset and target column. It then splits the dataset into training and
# testing sets, fits the classifier on the training data, and evaluates its performance on the testing data using the accuracy
# score. Finally, it prints the accuracy score and the target column for reference.
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled )
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Decision Tree Accuracy: {accuracy}')
print(f'Decision Tree F1 Score: {f1}')
# Save F1 score and string name
results['Decision Tree Classifier'] = {'accuracy': accuracy, 'f1_score': f1}


# K-FOLD CROSS VALIDATION 
# We are using logistic regression as the model to evaluate.
# The cross_val_score function will return an array of accuracy scores for each fold. You can then compute the mean and
# standard deviation of the scores to get an estimate of the model's accuracy on unseen data. 

# HYPERPARAMETER TUNING
# Perform hyperparameter tuning for a logistic regression model in scikit-learn using randomized search
# The hyperparameters being tuned are:
    # penalty: regularization penalty to be applied. It can be L1, L2, or ElasticNet.
    # C: inverse of regularization strength. Smaller values specify stronger regularization.
    # fit_intercept: whether to calculate the intercept for this model.
    # class_weight: weights associated with classes in the form of a dictionary, or ‘balanced’ to automatically adjust weights based on the class frequencies.
    # solver: algorithm to be used for optimization.
    # max_iter: maximum number of iterations taken for the solver to converge.
print("Hyperparameter Tuning: ")
print("TURN HYPERPARAMETER TUNING BACK ON")
'''
params = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'fit_intercept': [True, False],
    'class_weight': [None, 'balanced'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 500, 1000, 5000]
}

# The KFold function is used to split the data into k folds for cross-validation
# The n_iter parameter of RandomizedSearchCV is set to 100, which means that 100 random combinations of hyperparameters will be tried.
# n_jobs=-1 enables parallel processing using all available CPU cores.
cv = KFold(n_splits=4, shuffle=True, random_state=42)
model = LogisticRegression()
random_search = RandomizedSearchCV(model, param_distributions=params, cv=cv, n_iter=100, n_jobs=-1, random_state=42)
random_search.fit(X_train_scaled, y_train)
# predict on new data using the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
# The best hyperparameters found by the search are printed along with the best score achieved during cross-validation.
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_macro = f1_score(y_test, y_pred, average='macro')
print('Best parameters:', random_search.best_params_)
print('Best score:', random_search.best_score_)
print('Logistic Regression Accuracy score:', accuracy)
print('Logistic Regression F1 score:', f1)
results['Logistic Regression'] = {'accuracy': accuracy, 'f1_score': f1}
'''

# Print dictionary containing results
print('RESULTS')
print(results)

print(f"Count Readmitted: {df['readmitted'].value_counts()}")

#DECODES DATASET
# Decode the encoded values in each column and print the DataFrame with the decoded values
for column, encoder in label_encoders.items():
    df[column] = encoder.inverse_transform(df[column])

# GRAPHS
print('graphs')
OurGraphs.plot_pie_chart(df)
OurGraphs.plot_compare_f1_acc(results)

'''
OurGraphs.racemap(df)
OurGraphs.plot_race_countplot(df)
OurGraphs.plot_age_countplot(df)
OurGraphs.plot_gender_countplot(df)
OurGraphs.plot_diabetes_countplot(df)
OurGraphs.plot_hospital_countplot(df)
'''