import math
from GPyOpt.models.gpmodel import GPModel
import numpy as np
import logging
from graph_utils import get_random_graph

# This is the prior model
class MyGPModel(GPModel):
    def __init__(self, noise_var, exact_feval, optimize_restarts, exploration_weight_function, learnDependencyStructureRate, learnParameterRate, mlflow_logging, graph_function, fn, random_graph=False, additional_args=dict()):
        self.graph_function = graph_function
        self.fn = fn
        self.has_logged_inital = False
        #if random_graph:
        #    self.graph_function.graph = get_random_graph(self.graph_function.graph.number_of_nodes(), 1)
        _, kernel_full, self.cfn = graph_function.make_decomposition(self)
        self.t = 0
        self.exploration_weight_function = exploration_weight_function
        self.learnDependencyStructureRate = learnDependencyStructureRate
        if learnParameterRate == None:
            self.learnParameterRate = learnDependencyStructureRate
        else:
            self.learnParameterRate = learnParameterRate
        self.mlflow_logging = mlflow_logging
        super(MyGPModel, self).__init__(kernel=kernel_full, noise_var=noise_var, exact_feval=exact_feval, optimize_restarts=optimize_restarts)
        self.kernels_dict = self.graph_function.kernels
        self.random_graph = random_graph
        logging.info(f"Init graph{self.graph_function.graph.edges()}")

        self.additional_args = additional_args
    
    def predict(self, X, with_noise=True):
        raise Exception
    def predict_withGradients(self, X):
        raise Exception

    def predict_with_kernel(self, X, kernel):
        if X.ndim == 1:
            X = X[None,:]
        # self.model -> GPRegression
        #self.model.kern = kernel
        #m, v = self.model.predict(X, full_cov=False, include_likelihood=True)
        m, v = self.model.predict(X, kern=kernel, full_cov=False, include_likelihood=True)
        # Stability issues?
        v = np.clip(v, 1e-10, np.inf)
        # We can take the square root because v is just a diagonal matrix of variances
        return m, np.sqrt(v)

    def predict_withGradients_with_kernel(self, X, kernel):
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X, kern=kernel, full_cov=False, include_likelihood=True)
        v = np.clip(v, 1e-10, np.inf)

        dmdx, dvdx = self.model.predictive_gradients(X, kern=kernel)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))
        return m, np.sqrt(v), dmdx, dsdx


    def exploration_weight(self):
        return self.exploration_weight_function(self.t)
    
    # Update t when the model is updated
    def updateModel(self, X_all, Y_all, X_new, Y_new, log=True):
        super(MyGPModel, self).updateModel(X_all, Y_all, X_new, Y_new)

        if log:
            if self.t == 0:
                assert(len(self.fn.history_y) == len(Y_all))
                # Attempt to return f instead of y if that exist
                self.mlflow_logging.log_init_y(np.min(self.fn.history_y))
            else:
                assert(len(self.fn.history_y) == len(Y_all))
                self.mlflow_logging.log_y(np.min(self.fn.history_y[-1]))
        self.t+=1

    def update_structure(self, acquisition, X_all, Y_all, fn_optimizer, fn):

        if self.learnDependencyStructureRate < 0:
            return

        # Decide when to learn new structure
        if self.t % self.learnDependencyStructureRate == 0:
            if not self.has_logged_inital:
                # Log the inital graph
                fn.mlflow_logging.log_graph_metrics(self.graph_function.graph)
                self.has_logged_inital = True

            Y_vect = Y_all.flatten()
            if self.random_graph:
                dimsize = self.graph_function.graph.number_of_nodes()
                logging.info("Restarting graph")
                size_of_random_graph = self.additional_args.get('size_of_random_graph', 0.2)
                self.graph_function.graph = get_random_graph(dimsize, max(1,int(dimsize*size_of_random_graph)))
                #self.graph_function.dimensional_parameters = [(0.1, 0.5) for _ in range(self.graph_function.dimension())]
                fn_optimizer.optimize_parameters(X_all, Y_vect, self.graph_function)              
            else:
                fn_optimizer.optimize(X_all, Y_vect, self.graph_function)

            logging.info("New graph : {}".format(self.graph_function.graph.edges()))
            fn.mlflow_logging.log_graph_metrics(self.graph_function.graph)

            # Make acquisitions
            # ========================================================================
            # Update the decomposition used
            logging.debug("Dim Param: {}".format(self.graph_function.dimensional_parameters))
            _, self.kernel, self.cfn = self.graph_function.make_decomposition(self)
            self.kernels_dict = self.graph_function.kernels
            self.model = None 
        elif self.t % self.learnParameterRate == 0:
            # learn the parameters
            
            Y_vect = Y_all.flatten()
            fn_optimizer.optimize_parameters(X_all, Y_vect, self.graph_function)

            logging.debug("Dim Param: {}".format(self.graph_function.dimensional_parameters))
            _, self.kernel, self.cfn = self.graph_function.make_decomposition(self)
            self.model = None

            self.kernels_dict = self.graph_function.kernels