import pandas as pd

def load_data():
    # load data
    df = pd.read_csv('DataSets\Diabetes\diabetic_data.csv')

    # create DataFrame
    df = pd.DataFrame(df)

    # drop columns that are not relevant to the analysis
    df.drop(['diag_1','diag_2','diag_3','weight', 'medical_specialty', 'payer_code', 'encounter_id','patient_nbr', 'admission_type_id', 'discharge_disposition_id','admission_source_id'], axis=1, inplace=True)

    # replace '?' with 'Other' in the 'race' column
    df['race'] = df['race'].replace('?', 'Other')
    return df
