import numpy as np
import pandas as pd

class DataLogger:

    def __init__(self, size):
        self.df = self.res = pd.DataFrame(np.nan, index=np.arange(int(size) + 1),
                                          columns=['Index', 'LastValue',
                                                   'BestValue', 'Time',
                                                   'LastProtein',
                                                   'BestProtein'])
