import torch
import argparse
import pandas as pd
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.kernels import RBFKernel, AdditiveKernel

class Hartmann6:
    def __init__(self):
        self.alpha = [1.00, 1.20, 3.00, 3.20]
        self.A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                    [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                    [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                    [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        self.P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                            [2329, 4135, 8307, 3736, 1004, 9991],
                            [2348, 1451, 3522, 2883, 3047, 6650],
                            [4047, 8828, 8732, 5743, 1091, 381]])

    def objective_function(self, x, **kwargs):
        """6d Hartmann test function
            input bounds:  0 <= xi <= 1, i = 1..6
            global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
            min function value = -3.32237
        """

        external_sum = 0
        for i in range(4):
            internal_sum = 0
            for j in range(6):
                internal_sum += self.A[i, j] * (x[:,j] - self.P[i, j]) ** 2
            external_sum += self.alpha[i] * np.exp(-internal_sum)
        
        return external_sum.view(-1,1)
    
    def get_meta_information(self):
        return {'name': 'Hartmann6',
                'num_function_evals': 200,
                'optima': (
                [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]),
                'bounds': [[0, 1]] * 6,
                'f_opt': 3.322368011391339}


parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=300)
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

torch.manual_seed(args.seed)

bb = Hartmann6()
X = torch.rand(10, 6)
y = bb.objective_function(X)
fopt = bb.get_meta_information()['f_opt']

data = pd.DataFrame(columns=['best_regret', 'instant_regret'])
best_regret = float('inf')


for it in range(args.N):
    gp = SingleTaskGP(X, y, covar_module=AdditiveKernel(*[RBFKernel(active_dims=torch.tensor([i])) for i in range(6)]) if args.sparse else RBFKernel())
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll)
    except:
        print("Model fitting failed, exitting gracefully.")
        data.to_csv(f'hdbo/bo_results/sparse={args.sparse}seed={args.seed}.csv')
        continue

    UCB = UpperConfidenceBound(gp, beta=10)
    bounds = torch.stack([torch.zeros(6), torch.ones(6)])
    candidate, acq_value = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=40,
    )

    new_y = bb.objective_function(candidate)
    X = torch.cat([X, candidate], dim=0)
    y = torch.cat([y, new_y], dim=0)

    best_regret = min(best_regret, fopt - new_y)

    data = data.append(
        pd.DataFrame(
            {
                'best_regret': [best_regret.item()],
                'instant_regret': [fopt - new_y.item()] 
            },
            index = [it]
        )
    )
    print(it, best_regret.item(), candidate)

data.to_csv(f'hdbo/bo_results/sparse={args.sparse}seed={args.seed}.csv')
