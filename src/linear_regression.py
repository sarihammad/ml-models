import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.feature_selection import RFE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'processed')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'linear_regression')
MODEL_NAME = 'linear_regression_model.pkl'
PLOT_PATH = os.path.join(BASE_DIR, 'models', 'linear_regression')

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(PLOT_PATH):
    os.makedirs(PLOT_PATH)

def load_data(filename):
    """
    Load dataset from a CSV file.

    Args:
        filename (str): The name of the CSV file in the processed folder
    
    Returns:
        DataFrame: the loaded dataset
    """
    data = pd.read_csv(os.path.join(DATA_PATH, filename))
    print(f"Data loaded successfully with shape: {data.shape}")
    return data

def rfe_feature_selection(data, target_column, n_features=10):
    """
    Perform Recursive Feature Elimination (RFE) to select the best subset of features.

    Args:
        data (DataFrame): The dataset containing the features.
        target_column (str): The name of the target column.
        n_features (int): The number of features to select.

    Returns:
        selected_features (list): The column names of the selected subset.
    """
    print("Starting Recursive Feature Elimination (RFE)...")

    # Get only the integer and float columns
    integer_columns = data.select_dtypes(include=['int64', 'float64']).columns
    integer_columns = integer_columns.drop(target_column)
    print(f"Integer columns found: {list(integer_columns)}")

    # Split data
    X = data[integer_columns]
    y = data[target_column]

    # Remove rows with NaNs instead of filling
    X = X.dropna()
    y = y.loc[X.index]

    # Initialize the model and RFE
    model = LinearRegression()
    selector = RFE(model, n_features_to_select=n_features)
    selector = selector.fit(X, y)

    # Get the selected features
    selected_features = X.columns[selector.support_].tolist()
    print(f"[INFO] Selected features: {selected_features}")
    return selected_features

def preprocess_data(data, target_column):
    """
    Preprocess the dataset by splitting features and target variable,
    and then splitting into training and testing datasets.

    Args:
        data (DataFrame): the input dataset.
        target_column (str): the name of the target variable.
    Returns:
        Split datasets: X_train, X_test, y_train, y_test
    """
    print("Checking for missing values...")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])
    print(f"Total missing values: {missing_values.sum()}")

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    print("Missing values after filling:")
    print(data.isnull().sum().sum())

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.shape[0] == 0:
        raise ValueError("Error: No valid rows left after preprocessing. Check for missing values.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[Data preprocessed. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train the Linear Regression model using the training data.

    Args:
        X_train (DataFrame): training features
        y_train (Series): training target variable

    Returns:
        Trained Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Squared Error, Mean Absolute Error, and R^2 Score.

    Args:
        model (LinearRegression): trained model
        X_test (DataFrame): test features
        y_test (Series): test target variable
    """
    y_predict = model.predict(X_test)
    print("Model Evaluation Metrics:\n\n")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_predict):.4f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_predict):.4f}")
    print(f"R^2 Score: {r2_score(y_test, y_predict):.4f}")

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_predict, alpha=0.7, color='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_PATH, "linear_regression_plot.png"))
    plt.show()

def save_model(model):
    """
    Save the trained model to disk for future use.

    Args:
        model (LinearRegression): The trained model.
    """
    joblib.dump(model, os.path.join(MODEL_PATH, MODEL_NAME))
    print(f"Model saved to {os.path.join(MODEL_PATH, MODEL_NAME)}")



if __name__ == "__main__":
    # Load the data
    data = load_data('house_prices.csv')

    # Perform RFE to find the best subset of integer columns
    best_features = rfe_feature_selection(data, target_column='SalePrice', n_features=10)
    
    # Preprocess only the best features found
    data = data[best_features + ['SalePrice']]
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column='SalePrice')

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model
    save_model(model)