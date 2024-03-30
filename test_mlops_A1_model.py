import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from ..mlops_A1_model import main

def test_main_function(capsys):
    # Call the main function
    main()

    # Capture the printed output of the main function
    captured = capsys.readouterr()

    # Extract the accuracy from the printed output
    output_accuracy = float(captured.out.strip().split(":")[-1].strip())

    # Define the expected accuracy (you can adjust this value based on the actual output)
    expected_accuracy = 0.95

    # Assert that expected accuracy matches with output accuracy
    assert np.isclose(expected_accuracy, output_accuracy, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
