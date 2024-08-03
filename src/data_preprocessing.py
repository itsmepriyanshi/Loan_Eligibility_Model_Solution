import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(df):
    df.loc[:, 'Gender'] = df['Gender'].fillna('Male')
    df.loc[:, 'Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df.loc[:, 'Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df.loc[:, 'Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df.loc[:, 'LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df.loc[:, 'Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df.loc[:, 'Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    return df

def encode_categorical_variables(df):
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
    return df.drop('Loan_ID', axis=1)

def scale_data(xtrain, xtest):
    scaler = MinMaxScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    return xtrain_scaled, xtest_scaled
