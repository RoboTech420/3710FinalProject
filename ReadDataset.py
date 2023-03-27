import pandas as pd

def load_data():
    # load data
    df = pd.read_csv('DataSets\Diabetes\diabetic_data.csv')

    # create DataFrame
    df = pd.DataFrame(df)

    # replace '?' with 'Other' in the 'race' column
    df['race'] = df['race'].replace('?', 'Other')

    #start by setting all values containing E or V into 0 (as one category)
    df.loc[df['diag_1'].str.contains('V',na=False,case=False), 'diag_1'] = 0
    df.loc[df['diag_1'].str.contains('E',na=False,case=False), 'diag_1'] = 0
    df.loc[df['diag_2'].str.contains('V',na=False,case=False), 'diag_2'] = 0
    df.loc[df['diag_2'].str.contains('E',na=False,case=False), 'diag_2'] = 0
    df.loc[df['diag_3'].str.contains('V',na=False,case=False), 'diag_3'] = 0
    df.loc[df['diag_3'].str.contains('E',na=False,case=False), 'diag_3'] = 0

    #seting all missing values into -1
    df['diag_1'] = df['diag_1'].replace('?', -1)
    df['diag_2'] = df['diag_2'].replace('?', -1)
    df['diag_3'] = df['diag_3'].replace('?', -1)

    # Replace the unknown values
    df['gender'] = df['gender'].replace('Unknown/Invalid', 'Male')

    # Drop the duplicate patient records
    df = df.drop_duplicates(subset=['patient_nbr'])

    # drop columns that are not relevant to the analysis
    df.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'repaglinide', 'nateglinide', 'chlorpropamide', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'glyburide_metformin', 'glipizide_metformin', 'glimepiride_pioglitazone', 'metformin_rosiglitazone','metformin_pioglitazone', 'acetohexamide', 'tolbutamide'], axis=1, inplace=True)
    return df

def checkDataSet(df):
    print(df)
    print(df.isnull().sum())
    print(df[df.isnull().any(axis=1)])
    return

def printDatasetInfo(df):
    # Get the names of columns that contain "?"
    cols_with_questionmark = [col for col in df.columns if '?' in df[col].values]
    # Loop over each column in the DataFrame and print its value_counts()
    for col in df.columns:
        print("Column:", col)
        print(df[col].value_counts())
    return

#df = load_data()
#print(df.dtypes)
#checkDataSet(df)
#printDatasetInfo()
