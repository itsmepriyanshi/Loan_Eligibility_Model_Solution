# Credit Loan Prediction

This project aims to predict loan approval status based on various features using machine learning models. The dataset includes attributes like 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'LoanAmount', and 'Credit_History', and the target variable is 'Loan_Status'.

## Project Structure

The project is organized as follows:

- `main.py`: Entry point of the application. This script loads the dataset, processes it, trains models, evaluates them, and logs results.
- `src/data_preprocessing.py`: Contains functions for loading data, handling missing values, encoding categorical variables, and scaling data.
- `src/models.py`: Contains functions for training Logistic Regression and Random Forest models.
- `src/evaluation.py`: Contains functions for evaluating the performance of trained models.
- `src/utils.py`: Contains utility functions for logging setup and error logging.
- `logs/app.log`: Log file where application logs and errors are recorded.

## Requirements

Make sure to install the following Python packages to run this project:

- `pandas`
- `scikit-learn`

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
