import numpy as np
from data import make_data

def write_data(data, filename):
    np.save(filename, data)

def load_data(filename):
    return np.load(filename)

if __name__ == "__main__":
    SAMPLE_NUMBER = 2000
    X, y = make_data(n_samples=SAMPLE_NUMBER)

    write_data(X,"X_data")
    write_data(y, "y_data")