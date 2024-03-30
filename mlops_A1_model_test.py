import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlops_A1_model import main


def test_main_function():
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    expected_accuracy = accuracy_score(y_test, y_pred)

    # Call the main function
    main()

    # Capture the printed output of the main function
    captured = capsys.readouterr()

    # Extract accuracy from printed output
    lines = captured.out.strip().split("\n")
    last_line = lines[-1]
    output_accuracy = float(last_line.split(":")[-1].strip())

    # Assert that expected accuracy matches with output accuracy
    assert np.isclose(expected_accuracy, output_accuracy, atol=1e-5)


if __name__ == "_main_":
    pytest.main([_file_])