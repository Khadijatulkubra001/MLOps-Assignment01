import subprocess


def test_script_execution():
    try:
        subprocess.run(["python", "mlops_A1_model.py"], check=True)
    except subprocess.CalledProcessError:
        assert False, "Error occurred while running the script."


def test_accuracy():
    try:
        output = subprocess.check_output(["python", "mlops_A1_model.py"])
        accuracy = float(output.strip().split(b":")[1])
        assert 0 <= accuracy <= 1, "Accuracy value is out of range."
    except (subprocess.CalledProcessError, ValueError):
        assert False, "Error occurred while getting accuracy."


if __name__ == "__main__":
    test_script_execution()
    test_accuracy()
    print("All tests passed.")
