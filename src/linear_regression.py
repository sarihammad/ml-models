import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

DATA_PATH = '../datasets/processed/'
MODEL_PATH = '../models/linear_regression/'
MODEL_NAME = 'linear_regresion_model.pkl'
PLOT_PATH = '../models/linear_regression/'

def load_data(filename):
    """
    Load dataset from a CSV file.

    Args:
        filename (str): The name of the CSV file in the processed folder
    
    Returns:
        DataFrame: the loaded dataset
    """
    data = pd.read_csv(f"{DATA_PATH}{filename}")
    print(f"Data loaded successfully with shape: {data.shape}")
    return data

def preprocess_data(data, target_column):
    """
    Preprocess the dtaaset by splitting features and target variable,
    and then splitting into training and testing datasets.

    Args:
        data (DataFrame): the input dataset.
        target_column (str): the name of the target variable.
    Returns:
        Split datasets: X_train, X_test, y_train, y_test
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data preprocessed. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
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
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, predictions):.4f}")
    print(f"R^2 Score: {r2_score(y_test, y_predict):.4f}")

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_predict, alpha=0.7, color='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.grid(True)
    plt.savefig(f"{PLOT_PATH}linear_regression_plot.png")
    plt.show()

def save_model(model):
    """
    Save the trained model to disk for future use.

    Args:
        model (LinearRegression): The trained model.
    """
    joblib.dump(model, f"{MODEL_PATH}{MODEL_NAME}")
    print(f"Model saved to {MODEL_PATH}{MODEL_NAME}")

if __name__ == "__main__":
    data = load_data('house_prices.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column='price')
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
