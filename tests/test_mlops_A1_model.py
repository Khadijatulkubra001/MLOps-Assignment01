import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ..mlops_A1_model import main


def test_main_function(capsys):
    # Mocking the iris dataset
    X_train_mock = np.array([[5.1, 3.5, 1.4, 0.2],
                              [4.9, 3.0, 1.4, 0.2],
                              [4.7, 3.2, 1.3, 0.2],
                              [4.6, 3.1, 1.5, 0.2],
                              [5.0, 3.6, 1.4, 0.2],
                              [7.0, 3.2, 4.7, 1.4],
                              [6.4, 3.2, 4.5, 1.5],
                              [6.9, 3.1, 4.9, 1.5],
                              [5.5, 2.3, 4.0, 1.3],
                              [6.3, 3.3, 6.0, 2.5],
                              [5.8, 2.7, 5.1, 1.9],
                              [7.1, 3.0, 5.9, 2.1],
                              [6.3, 2.9, 5.6, 1.8],
                              [6.5, 3.0, 5.8, 2.2],
                              [7.6, 3.0, 6.6, 2.1]])
    y_train_mock = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    X_test_mock = np.array([[6.4, 2.8, 5.6, 2.1],
                             [5.7, 2.8, 4.5, 1.3],
                             [6.7, 3.1, 4.7, 1.5],
                             [6.0, 3.4, 4.5, 1.6],
                             [6.1, 2.6, 5.6, 1.4]])
    y_test_mock = np.array([2, 1, 1, 1, 2])

    # Save the original iris dataset to restore later
    original_load_iris = load_iris

    # Mocking the load_iris function
    def mocked_load_iris():
        class MockIris:
            def _init_(self):
                self.data = X_train_mock
                self.target = y_train_mock

        return MockIris()

    # Replace load_iris with mocked_load_iris
    load_iris = mocked_load_iris

    # Call the main function
    main()

    # Restore the original load_iris function
    load_iris = original_load_iris

    # Capture the printed output of the main function
    captured = capsys.readouterr()

    # Extract the accuracy from the printed output
    output_accuracy = float(captured.out.strip().split(":")[-1].strip())

    # Fit a logistic regression model on the mock data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_mock)
    X_test_scaled = scaler.transform(X_test_mock)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train_mock)

    # Predict on the mock test set
    y_pred_mock = model.predict(X_test_scaled)

    # Calculate the accuracy
    expected_accuracy = accuracy_score(y_test_mock, y_pred_mock)

    # Assert that expected accuracy matches with output accuracy
    assert np.isclose(expected_accuracy, output_accuracy, atol=1e-5)


if _name_ == "_main_":
    pytest.main([_file_])
