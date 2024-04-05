import numpy as np
from .utils import compute_gaussian_product_coef, unsqueeze_xdim
from functools import partial
import torch
from tqdm import tqdm

class Scheduler():
    def __init__(self, betas, device):

        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode: xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def ddpm_sampling(self, steps, pred_x0_fn, x1, ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)

        xs = [xt.cpu()]
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
    
    def exp_sampling(self, steps, pred_eps_fn, x1, log_steps=None, ab_order = 0):
        import deis.th_deis as deis
        import jax.numpy as jnp

        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        num_steps = len(steps) - 1
        
        class VESDE:
            def __init__(self, betas, std_fwd, std_bwd, sampling_eps = 0, sampling_T = 999):
                self._sampling_eps = sampling_eps
                self._sampling_T = sampling_T
                betas = jnp.array(betas.cpu().numpy())
                std_fwd = jnp.array(std_fwd.cpu().numpy())
                std_bwd = jnp.array(std_bwd.cpu().numpy())
                j_times = jnp.asarray(
                    jnp.arange(len(betas)), dtype=float
                )
                self.betas_interpol = deis.vpsde.get_interp_fn(j_times, betas)
                self.std_fwd_interpol = deis.vpsde.get_interp_fn(j_times, std_fwd)
                self.std_bwd_interpol = deis.vpsde.get_interp_fn(j_times, std_bwd)

            @property
            def is_continuous(self):
                return False
            @property
            def sampling_T(self):
                return self._sampling_T

            @property
            def sampling_eps(self):
                return self._sampling_eps

            def psi(self, t_start, t_end):
                return jnp.ones_like(t_start)
            
            def eps_integrand(self, vec_t):
                integrand = self.betas_interpol(vec_t)/self.std_fwd_interpol(vec_t)
                return integrand

        sde = VESDE(self.betas, self.std_fwd, self.std_bwd)
        sampler_fn = deis.get_sampler(
            # args for diffusion model
            sde,
            pred_eps_fn,
            # args for timestamps scheduling
            ts_phase="t", # support "rho", "t", "log"
            ts_order=1.0,
            num_step=num_steps,
            # deis choice
            method = "t_ab", # deis sampling algorithms: support "rho_rk", "rho_ab", "t_ab", "ipndm"
            ab_order= ab_order, # greater than 0, used for "rho_ab", "t_ab" algorithms, other algorithms will ignore the arg
            rk_method="3kutta" # used for "rho_rk" algorithms, other algorithms will ignore the arg
        )
        xs = sampler_fn(xt)
        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), None

    def compute_label(self, step, x0, xt):
        std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()