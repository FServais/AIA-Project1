from sklearn.utils import check_random_state
from data import make_data


def get_random_state():
    return check_random_state(0)

def get_dataset(sample_number):
    random_state = get_random_state()
    X, y = make_data(sample_number, random_state=random_state)
    return X, y, random_state