# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import numpy as np
from torch import Tensor
from torch.distributions import Normal
from abc import ABC, abstractmethod
from ..models.base_model import BaseModel

class Acquisition(ABC):
    def __init__(self, model, **conf):
        self.model = model

    @property
    @abstractmethod
    def num_obj(self):
        pass

    @property
    @abstractmethod
    def num_constr(self):
        pass

    @abstractmethod
    def eval(self, x : Tensor,  xe : Tensor) -> Tensor:
        """
        Shape of output tensor: (x.shape[0], self.num_obj + self.num_constr)
        """
        pass

    def __call__(self, x : Tensor,  xe : Tensor):
        return self.eval(x, xe)

class SingleObjectiveAcq(Acquisition):
    """
    Single-objective, unconstrained acquisition
    """
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)

    @property
    def num_obj(self):
        return 1

    @property
    def num_constr(self):
        return 0

class LCB(SingleObjectiveAcq):
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get('kappa', 3.0)
        assert(model.num_out == 1)
    
    def eval(self, x : Tensor, xe : Tensor) -> Tensor:
        py, ps2 = self.model.predict(x, xe)
        return py - self.kappa * ps2.sqrt()

class Mean(SingleObjectiveAcq):
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)
        assert(model.num_out == 1)

    def eval(self, x : Tensor, xe : Tensor) -> Tensor:
        py, _ = self.model.predict(x, xe)
        return py

class Sigma(SingleObjectiveAcq):
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)
        assert(model.num_out == 1)

    def eval(self, x : Tensor, xe : Tensor) -> Tensor:
        _, ps2 = self.model.predict(x, xe)
        return -1 * ps2.sqrt()

class EI(SingleObjectiveAcq):
    pass

class logEI(SingleObjectiveAcq):
    pass

class WEI(Acquisition):
    pass

class Log_WEI(Acquisition):
    pass

class MES(SingleObjectiveAcq):
    pass

class MOMeanSigmaLCB(Acquisition):
    def __init__(self, model, best_y, **conf):
        super().__init__(model, **conf)
        self.best_y = best_y
        self.kappa  = conf.get('kappa', 2.0)
        assert(self.model.num_out == 1)

    @property
    def num_obj(self):
        return 2

    @property
    def num_constr(self):
        return 1

    def eval(self, x: Tensor, xe : Tensor) -> Tensor:
        """
        minimize (py, -1 * ps)
        s.t.     LCB  < best_y
        """
        with torch.no_grad():
            out        = torch.zeros(x.shape[0], self.num_obj + self.num_constr)
            py, ps2    = self.model.predict(x, xe)
            noise      = np.sqrt(self.model.noise)
            py        += noise * torch.randn(py.shape)
            ps         = ps2.sqrt()
            lcb        = py - self.kappa * ps
            out[:, 0]  = py.squeeze()
            out[:, 1]  = -1 * ps.squeeze()
            out[:, 2]  = lcb.squeeze() - self.best_y # lcb - best_y < 0
            return out

class MACE(Acquisition):
    def __init__(self, model, best_y, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get('kappa', 2.0)
        self.eps   = conf.get('eps', 1e-4)
        self.tau   = best_y
    
    @property
    def num_constr(self):
        return 0

    @property
    def num_obj(self):
        return 3

    def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
        """
        minimize (-1 * EI,  -1 * PI, lcb)
        """
        with torch.no_grad():
            py, ps2   = self.model.predict(x, xe)
            noise     = np.sqrt(2.0) * self.model.noise.sqrt()
            ps        = ps2.sqrt().clamp(min = torch.finfo(ps2.dtype).eps)
            lcb       = (py + noise * torch.randn(py.shape)) - self.kappa * ps
            normed    = ((self.tau - self.eps - py - noise * torch.randn(py.shape)) / ps)
            dist      = Normal(0., 1.)
            log_phi   = dist.log_prob(normed)
            Phi       = dist.cdf(normed)
            PI        = Phi
            EI        = ps * (Phi * normed +  log_phi.exp())
            logEIapp  = ps.log() - 0.5 * normed**2 - (normed**2 - 1).log()
            logPIapp  = -0.5 * normed**2 - torch.log(-1 * normed) - torch.log(torch.sqrt(torch.tensor(2 * np.pi)))

            use_app             = ~((normed > -6) & torch.isfinite(EI.log()) & torch.isfinite(PI.log())).reshape(-1)
            out                 = torch.zeros(x.shape[0], 3)
            out[:, 0]           = lcb.reshape(-1)
            out[:, 1][use_app]  = -1 * logEIapp[use_app].reshape(-1)
            out[:, 2][use_app]  = -1 * logPIapp[use_app].reshape(-1)
            out[:, 1][~use_app] = -1 * EI[~use_app].log().reshape(-1)
            out[:, 2][~use_app] = -1 * PI[~use_app].log().reshape(-1)
            return out

class NoisyAcq(Acquisition):
    def __init__(self, model, num_obj, num_constr):
        super().__init__(model)
        self._num_obj    = num_obj
        self._num_constr = num_constr

    @property
    def num_obj(self) -> int:
        return self._num_obj

    @property
    def num_constr(self) -> int:
        return self._num_constr

    def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
        with torch.no_grad():
            py, ps2 = self.model.predict(x, xe)
            y_samp  = py + ps2.sqrt() * torch.randn(py.shape)
            return y_samp

class GeneralAcq(Acquisition):
    def __init__(self, model, num_obj, num_constr, **conf):
        super().__init__(model, **conf)
        self._num_obj    = num_obj
        self._num_constr = num_constr
        self.kappa       = conf.get('kappa', 2.0)
        self.c_kappa     = conf.get('c_kappa', 0.)
        self.use_noise   = conf.get('use_noise', True)
        assert self.model.num_out == self.num_obj + self.num_constr
        assert self.num_obj >= 1

    @property
    def num_obj(self) -> int:
        return self._num_obj

    @property
    def num_constr(self) -> int:
        return self._num_constr

    def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
        """
        Acquisition function to deal with general constrained, multi-objective optimization problems
        
        Suppose we have $om$ objectives and $cn$ constraints, the problem should has been transformed to :

        Minimize (o1, o1, \dots,  om)
        S.t.     c1 < 0, 
                 c2 < 0, 
                 \dots
                 cb_cn < 0

        In this `GeneralAcq` acquisition function, we calculate lower
        confidence bound of objectives and constraints, and solve the following
        problem:

        Minimize (lcb_o1, lcb_o2, \dots,  lcb_om)
        S.t.     lcb_c1 < 0, 
                 lcb_c2 < 0, 
                 \dots
                 lcb_cn < 0
        """
        with torch.no_grad():
            py, ps2 = self.model.predict(x, xe)
            ps      = ps2.sqrt().clamp(min = torch.finfo(ps2.dtype).eps)
            if self.use_noise:
                noise  = self.model.noise.sqrt()
                py    += noise * torch.randn(py.shape)
            out = torch.ones(py.shape)
            out[:, :self.num_obj] = py[:, :self.num_obj]  - self.kappa   * ps[:, :self.num_obj]
            out[:, self.num_obj:] = py[:, self.num_obj:]  - self.c_kappa * ps[:, self.num_obj:]
        return out
