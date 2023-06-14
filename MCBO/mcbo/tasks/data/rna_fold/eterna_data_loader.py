from pathlib import Path

import pandas as pd
import os


def load_eterna_data() -> pd.DataFrame:
    path = os.path.join(str(Path(os.path.realpath(__file__)).parent), "eterna.csv")
    return pd.read_csv(path, index_col=0)
