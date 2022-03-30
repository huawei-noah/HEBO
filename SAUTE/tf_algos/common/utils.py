import random
import numpy as np
import tensorflow as tf

def set_random_seed(seed:int):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(0) 
    tf.random.set_seed(seed)