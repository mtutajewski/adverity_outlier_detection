import os

import pandas as pd
from pytest import *

DATA_DIR = "data/avazu-ctr-prediction"


def check_data_exists_test():
    assert os.path.exists(f"{DATA_DIR}/train.csv") == True


def verify_data_shape():
    df = pd.read_csv(f"../{DATA_DIR}/train.csv")
    assert df.shape[0] == 40428967
