from src.data_preprocessing import load_data, handle_missing_values, encode_categorical_variables, scale_data
from src.models import train_logistic_regression, train_random_forest
from src.evaluation import evaluate_model
from src.utils import setup_logging, log_error
from sklearn.model_selection import train_test_split

def main():
    setup_logging()
    try:
        df = load_data('data/credit.csv')
        print("Columns in the dataset:", df.columns)  # Print column names to check

        df = handle_missing_values(df)
        df = encode_categorical_variables(df)
        
        if 'Loan_Status' in df.columns:
            X = df.drop('Loan_Status', axis=1)
            y = df['Loan_Status']
        else:
            raise KeyError("Column 'Loan_Status' not found in the dataset.")
        
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=123)
        xtrain_scaled, xtest_scaled = scale_data(xtrain, xtest)
        
        print("Training Logistic Regression model...")
        lr_model = train_logistic_regression(xtrain_scaled, ytrain)
        evaluate_model(lr_model, xtrain_scaled, ytrain, xtest_scaled, ytest)
        
        print("Training Random Forest model...")
        rf_model = train_random_forest(xtrain_scaled, ytrain)
        evaluate_model(rf_model, xtrain_scaled, ytrain, xtest_scaled, ytest)
        
    except Exception as e:
        log_error(str(e))
        raise

if __name__ == "__main__":
    main()
