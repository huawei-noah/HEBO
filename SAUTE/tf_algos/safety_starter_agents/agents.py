"""
Agents copied from safety starter agents and modified.
"""
from safe_rl.pg.agents import Agent
from safe_rl.utils.mpi_tools import mpi_avg
import numpy as np
from safe_rl.pg.utils import EPS
import safe_rl.pg.trust_region as tro

class PPOAgent(Agent):
    
    def __init__(self, clip_ratio=0.2, 
                       pi_lr=3e-4, 
                       pi_iters=80, 
                       kl_margin=1.2,
                       **kwargs):
        super().__init__(**kwargs)
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.pi_iters = pi_iters
        self.kl_margin = kl_margin
        self.params.update(dict(
            clipped_adv=True,
            first_order=True,
            constrained=False
            ))

    def update_pi(self, inputs):

        # Things we need from training package
        train_pi = self.training_package['train_pi']
        d_kl = self.training_package['d_kl']
        target_kl = self.training_package['target_kl']

        # Run the update
        for i in range(self.pi_iters):
            _, kl = self.sess.run([train_pi, d_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > self.kl_margin * target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        self.logger.store(StopIter=i)

    def log(self):
        self.logger.log_tabular('StopIter', average_only=True)

class TrustRegionAgent(Agent):

    def __init__(self, damping_coeff=0.1, 
                       backtrack_coeff=0.8, 
                       backtrack_iters=10, 
                       **kwargs):
        super().__init__(**kwargs)
        self.damping_coeff = damping_coeff
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters
        self.params.update(dict(
            trust_region=True
            ))

class TRPOAgent(TrustRegionAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params.update(dict(
            constrained=False
            ))

    def update_pi(self, inputs):

        flat_g = self.training_package['flat_g']
        v_ph = self.training_package['v_ph']
        hvp = self.training_package['hvp']
        get_pi_params = self.training_package['get_pi_params']
        set_pi_params = self.training_package['set_pi_params']
        pi_loss = self.training_package['pi_loss']
        d_kl = self.training_package['d_kl']
        target_kl = self.training_package['target_kl']

        Hx = lambda x : mpi_avg(self.sess.run(hvp, feed_dict={**inputs, v_ph: x}))
        g, pi_l_old = self.sess.run([flat_g, pi_loss], feed_dict=inputs)
        g, pi_l_old = mpi_avg(g), mpi_avg(pi_l_old)

        # Core calculations for TRPO or NPG
        x = tro.cg(Hx, g)
        alpha = np.sqrt(2*target_kl/(np.dot(x, Hx(x))+EPS))
        old_params = self.sess.run(get_pi_params)

        # Save lagrange multiplier
        self.logger.store(Alpha=alpha)

        def set_and_eval(step):
            self.sess.run(set_pi_params, feed_dict={v_ph: old_params - alpha * x * step})
            return mpi_avg(self.sess.run([d_kl, pi_loss], feed_dict=inputs))

        # TRPO augments NPG with backtracking line search, hard kl constraint
        for j in range(self.backtrack_iters):
            kl, pi_l_new = set_and_eval(step=self.backtrack_coeff**j)
            if kl <= target_kl and pi_l_new <= pi_l_old:
                self.logger.log('Accepting new params at step %d of line search.'%j)
                self.logger.store(BacktrackIters=j)
                break

            if j==self.backtrack_iters-1:
                self.logger.log('Line search failed! Keeping old params.')
                self.logger.store(BacktrackIters=j)
                kl, pi_l_new = set_and_eval(step=0.)

    def log(self):
        self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('BacktrackIters', average_only=True)

class CPOAgent(TrustRegionAgent):

    def __init__(self, learn_margin=False, **kwargs):
        super().__init__(**kwargs)
        self.learn_margin = learn_margin
        self.params.update(dict(
            constrained=True,
            save_penalty=True
            ))
        self.margin = 0
        self.margin_lr = 0.05


    def update_pi(self, inputs):

        flat_g = self.training_package['flat_g']
        flat_b = self.training_package['flat_b']
        v_ph = self.training_package['v_ph']
        hvp = self.training_package['hvp']
        get_pi_params = self.training_package['get_pi_params']
        set_pi_params = self.training_package['set_pi_params']
        pi_loss = self.training_package['pi_loss']
        surr_cost = self.training_package['surr_cost']
        d_kl = self.training_package['d_kl']
        target_kl = self.training_package['target_kl']
        cost_lim = self.training_package['cost_lim']

        Hx = lambda x : mpi_avg(self.sess.run(hvp, feed_dict={**inputs, v_ph: x}))
        outs = self.sess.run([flat_g, flat_b, pi_loss, surr_cost], feed_dict=inputs)
        outs = [mpi_avg(out) for out in outs]
        g, b, pi_l_old, surr_cost_old = outs


        # Need old params, old policy cost gap (epcost - limit), 
        # and surr_cost rescale factor (equal to average eplen).
        old_params = self.sess.run(get_pi_params)
        c = self.logger.get_stats('EpCost')[0] - cost_lim
        rescale = self.logger.get_stats('EpLen')[0]

        # Consider the right margin
        if self.learn_margin:
            self.margin += self.margin_lr * c
            self.margin = max(0, self.margin)

        # The margin should be the same across processes anyhow, but let's
        # mpi_avg it just to be 100% sure there's no drift. :)
        self.margin = mpi_avg(self.margin)

        # Adapt threshold with margin.
        c += self.margin

        # c + rescale * b^T (theta - theta_k) <= 0, equiv c/rescale + b^T(...)
        c /= (rescale + EPS)

        # Core calculations for CPO
        v = tro.cg(Hx, g)
        approx_g = Hx(v)
        q = np.dot(v, approx_g)

        # Determine optim_case (switch condition for calculation,
        # based on geometry of constrained optimization problem)
        if np.dot(b,b) <= 1e-8 and c < 0:
            # feasible and cost grad is zero---shortcut to pure TRPO update!
            w, r, s, A, B = 0, 0, 0, 0, 0
            optim_case = 4
        else:
            # cost grad is nonzero: CPO update!
            w = tro.cg(Hx, b)
            r = np.dot(w, approx_g)         # b^T H^{-1} g
            s = np.dot(w, Hx(w))            # b^T H^{-1} b
            A = q - r**2 / s                # should be always positive (Cauchy-Shwarz)
            B = 2*target_kl - c**2 / s      # does safety boundary intersect trust region? (positive = yes)

            if c < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif c < 0 and B >= 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif c >= 0 and B >= 0:
                # x = 0 is infeasible and safety boundary intersects
                # ==> part of trust region is feasible, recovery possible
                optim_case = 1
                self.logger.log('Alert! Attempting feasible recovery!', 'yellow')
            else:
                # x = 0 infeasible, and safety halfspace is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                self.logger.log('Alert! Attempting infeasible recovery!', 'red')

        if optim_case in [3,4]:
            lam = np.sqrt(q / (2*target_kl))
            nu = 0
        elif optim_case in [1,2]:
            LA, LB = [0, r /c], [r/c, np.inf]
            LA, LB = (LA, LB) if c < 0 else (LB, LA)
            proj = lambda x, L : max(L[0], min(L[1], x))
            lam_a = proj(np.sqrt(A/B), LA)
            lam_b = proj(np.sqrt(q/(2*target_kl)), LB)
            f_a = lambda lam : -0.5 * (A / (lam+EPS) + B * lam) - r*c/(s+EPS)
            f_b = lambda lam : -0.5 * (q / (lam+EPS) + 2 * target_kl * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * c - r) / (s + EPS)
        else:
            lam = 0
            nu = np.sqrt(2 * target_kl / (s+EPS))

        # normal step if optim_case > 0, but for optim_case =0,
        # perform infeasible recovery: step to purely decrease cost
        x = (1./(lam+EPS)) * (v + nu * w) if optim_case > 0 else nu * w

        # save intermediates for diagnostic purposes
        self.logger.store(Optim_A=A, Optim_B=B, Optim_c=c,
                          Optim_q=q, Optim_r=r, Optim_s=s,
                          Optim_Lam=lam, Optim_Nu=nu, 
                          Penalty=nu, DeltaPenalty=0,
                          Margin=self.margin,
                          OptimCase=optim_case)

        def set_and_eval(step):
            self.sess.run(set_pi_params, feed_dict={v_ph: old_params - step * x})
            return mpi_avg(self.sess.run([d_kl, pi_loss, surr_cost], feed_dict=inputs))

        # CPO uses backtracking linesearch to enforce constraints
        self.logger.log('surr_cost_old %.3f'%surr_cost_old, 'blue')
        for j in range(self.backtrack_iters):
            kl, pi_l_new, surr_cost_new = set_and_eval(step=self.backtrack_coeff**j)
            self.logger.log('%d \tkl %.3f \tsurr_cost_new %.3f'%(j, kl, surr_cost_new), 'blue')
            if (kl <= target_kl and
                (pi_l_new <= pi_l_old if optim_case > 1 else True) and
                surr_cost_new - surr_cost_old <= max(-c,0)):
                self.logger.log('Accepting new params at step %d of line search.'%j)
                self.logger.store(BacktrackIters=j)
                break

            if j==self.backtrack_iters-1:
                self.logger.log('Line search failed! Keeping old params.')
                self.logger.store(BacktrackIters=j)
                kl, pi_l_new, surr_cost_new = set_and_eval(step=0.)


    def log(self):
        self.logger.log_tabular('Optim_A', average_only=True)
        self.logger.log_tabular('Optim_B', average_only=True)
        self.logger.log_tabular('Optim_c', average_only=True)
        self.logger.log_tabular('Optim_q', average_only=True)
        self.logger.log_tabular('Optim_r', average_only=True)
        self.logger.log_tabular('Optim_s', average_only=True)
        self.logger.log_tabular('Optim_Lam', average_only=True)
        self.logger.log_tabular('Optim_Nu', average_only=True)
        self.logger.log_tabular('OptimCase', average_only=True)
        self.logger.log_tabular('Margin', average_only=True)
        self.logger.log_tabular('BacktrackIters', average_only=True)