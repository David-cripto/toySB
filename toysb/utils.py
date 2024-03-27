import torch as th
from torch.optim import AdamW, lr_scheduler
from .scheduler import Scheduler
import torch.nn.functional as F

def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = (
        th.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=th.float64) ** 2
    )
    return betas.numpy()

def build_optimizer_sched(opt, net, log):
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    return optimizer, sched

def train(opt, net, scheduler: Scheduler, train_dataloader, test_dataloader):
    net.train()

    for it in range(opt.num_epoch):
        optimizer.zero_grad()

        for x0, x1 in train_dataloader:
            step = th.randint(0, opt.interval, (x0.shape[0],))
            xt = scheduler.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
            eps_true = scheduler.compute_label(step, x0, xt)
            eps_pred = net(xt, step)

            loss = F.mse_loss(eps_pred, eps_true)
            loss.backward()
