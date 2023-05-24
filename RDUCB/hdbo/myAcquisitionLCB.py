from collections import defaultdict
import math
from GPyOpt.acquisitions import AcquisitionLCB
import numpy as np
import mlflow_logging

class MyAcquisitionLCB(AcquisitionLCB):
    def __init__(self, model, kernel, variables):
        super(MyAcquisitionLCB, self).__init__(model=model, space=None, exploration_weight=None)
        self.dimension = kernel.input_dim
        self.kernel = kernel
        self.variables = variables

    # Special computation to compute the exact kernel ad it relies on the model
    def _compute_acq(self, x):
        m, s = self.model.predict_with_kernel(x, self.kernel) 
        f_acqu = -m + self.model.exploration_weight() * s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        m, s, dmdx, dsdx = self.model.predict_withGradients_with_kernel(x, self.kernel)
        
        f_acqu = -m + self.model.exploration_weight() * s       
        df_acqu = -dmdx + self.model.exploration_weight() * dsdx
        return f_acqu, df_acqu

    # We override the default because of potential issues 
    def acquisition_function(self, x):
        f_acqu = self._compute_acq(x)
        return -f_acqu

    def acquisition_function_withGradients(self, x):
        f_acqu,df_acqu = self._compute_acq_withGradients(x)
        return -f_acqu, -df_acqu

    def __call__(self, x):
        return self.acquisition_function(x)