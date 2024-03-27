import torch as th
from torch.optim import AdamW, lr_scheduler
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
import os
from torch.utils.tensorboard import SummaryWriter

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

def create_symmetric_beta_schedule(n_timestep, *args, **kwargs):
    import numpy as np
    betas = make_beta_schedule(n_timestep=n_timestep, *args, **kwargs)
    betas = np.concatenate([betas[:n_timestep//2], np.flip(betas[:n_timestep//2])])
    return betas

def build_optimizer_sched(opt, net, logger):
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    logger.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        logger.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    return optimizer, sched

def sample_batch(opt, loader):
    x0, x1= next(loader)
    x0 = x0.detach().to(opt.device)
    x1 = x1.detach().to(opt.device)

    assert x0.shape == x1.shape

    return x0, x1

class TensorBoardWriter:
    def __init__(self, opt):
        run_dir = str(opt.log_dir / opt.name)
        os.makedirs(run_dir, exist_ok=True)
        self.writer=SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        self.writer.add_scalar(key, val, global_step=global_step)

    def close(self):
        self.writer.close()

def train(opt, net, scheduler, train_dataloader, logger):
    writer = TensorBoardWriter(opt)
    ema = ExponentialMovingAverage(net.parameters(), decay=opt.ema)
    noise_levels = th.linspace(opt.t0, opt.T, opt.num_steps, device=opt.device) * opt.num_steps

    ema.to(opt.device)
    net.to(opt.device)

    optimizer, sched = build_optimizer_sched(opt, net, logger)

    net.train()

    for it in range(opt.num_epoch):
        for x0, x1 in train_dataloader:
            optimizer.zero_grad()
            x0, x1 = x0.detach().to(opt.device), x1.detach().to(opt.device)
            step = th.randint(0, opt.num_steps, (x0.shape[0],))
            xt = scheduler.q_sample(step, x0, x1)
            eps_true = scheduler.compute_label(step, x0, xt)
            eps_pred = net(xt, noise_levels[step].detach())

            loss = F.mse_loss(eps_pred, eps_true)
            loss.backward()
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

        logger.info("train_it {}/{} | lr:{} | loss:{}".format(
                    1+it,
                    opt.num_epoch,
                    "{:.2e}".format(optimizer.param_groups[0]['lr']),
                    "{:+.4f}".format(loss.item()),
                ))
        writer.add_scalar(it, 'loss', loss.detach())
        th.save({
                    "net": net.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sched": sched.state_dict() if sched is not None else sched,
                }, opt.ckpt_path / "latest.pt")
        logger.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
    writer.close()