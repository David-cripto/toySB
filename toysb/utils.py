import torch as th
from torch.optim import AdamW, lr_scheduler
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_CONSTANTS = {
    "scale":1,
    "width":0.002,
    "x_range":(-15,15),
    "y_range":(-15,15),
    "figsize":(20,10)
}

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

def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps

class TensorBoardWriter:
    def __init__(self, opt):
        run_dir = str(opt.log_dir / opt.name)
        os.makedirs(run_dir, exist_ok=True)
        self.writer=SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        self.writer.add_scalar(key, val, global_step=global_step)

    def add_figure(self, global_step, key, val):
        self.writer.add_figure(key, val, global_step=global_step)

    def close(self):
        self.writer.close()

def compute_pred_x0(step, xt, net_out, scheduler):
    std_fwd = scheduler.get_std_fwd(step, xdim=xt.shape[1:])
    pred_x0 = xt - std_fwd * net_out
    return pred_x0

def visualize(xs, x0, log_steps):
    fig, axs = plt.subplots(1, xs.shape[1] + 1, figsize = (20, 10))
    
    axs[0].scatter(x0[:, 0], x0[:, 1], c = list(range(len(x0))))
    axs[0].set_title(f"True labels")

    for ind in range(1, xs.shape[1] + 1):
        points_t = xs[:, ind - 1, :]
        axs[ind].scatter(points_t[:, 0], points_t[:, 1], c = list(range(len(points_t))))
        axs[ind].set_title(f"Points at time {log_steps[ind - 1]}")

    return fig

def save_imgs(xs, log_steps, path_to_save, velocity_dict):
    colors = list(range(len(xs)))

    path_to_save = Path(path_to_save)

    for ind in range(xs.shape[1]):
        plt.figure(figsize = IMAGE_CONSTANTS["figsize"])
        points_t = xs[:, ind, :]
        plt.scatter(points_t[:, 0], points_t[:, 1], c = colors)
        plt.title(f"Points at time {log_steps[ind]}")
        if ind != 0:
            for name, vel in velocity_dict.items():
                plt.quiver(points_t[:, 0], points_t[:, 1], vel["vel"][:, ind - 1, 0], vel["vel"][:, ind - 1, 1], 
                           angles='xy', scale_units='xy', scale=IMAGE_CONSTANTS["scale"], 
                           label = name, width =IMAGE_CONSTANTS["width"], color = vel["color"],
                           alpha = 0.3)
        
        plt.legend()
        plt.xlim(*IMAGE_CONSTANTS["x_range"])
        plt.ylim(*IMAGE_CONSTANTS["y_range"])
        plt.savefig(str(path_to_save / f"{log_steps[ind]}.png"))

@th.no_grad()
def sampling(opt, val_dataloader, net, ema, scheduler, path_to_save = None):
    steps = th.arange(opt.num_steps).tolist()
    log_steps = space_indices(opt.num_steps, opt.log_count)

    for x0, x1 in val_dataloader:
        break

    x1 = x1.detach().to(opt.device)
    
    velocity_dict = {}
    
    with ema.average_parameters():
        net.eval()

        def pred_x0_fn(xt, step):
            step = th.full((xt.shape[0],), step, device=opt.device, dtype=th.long)
            out = net(xt, step)
            return compute_pred_x0(step, xt, out, scheduler)
        xs, pred_x0 = scheduler.ddpm_sampling(steps, pred_x0_fn, x1, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=True)
        if opt.vel:
            velocity_dict["True velocity"] = {"vel" : xs[:, :-1, :] - xs[:, 1:, :], "color" : "#BB8FCE"}
        if opt.exp_int_vel:
            def pred_eps_fn(xt, scalar_t):
                vec_t = th.full((xt.shape[0],), scalar_t, device=xt.device, dtype=th.long)
                out = net(xt, vec_t)
                return out
            exp_int_xs, pred_x0 = scheduler.exp_sampling(steps, pred_eps_fn, x1, log_steps=log_steps)

            velocity_dict["Exponential integrator velocity"] = {"vel" : exp_int_xs[:, :-1, :] - exp_int_xs[:, 1:, :], "color" : "#CA6F1E"}

    if path_to_save is not None:
        save_imgs(xs, log_steps, path_to_save, velocity_dict)

    figure = visualize(xs, x0, log_steps)

    return figure
    

def train(opt, net, scheduler, train_dataloader, val_dataloader, logger):
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
        if it % 5 == 0:
            writer.add_scalar(it, 'loss', loss.detach())

        if it % 10 == 0: # 0, 0.5k, 3k, 6k 9k
            net.eval()
            logger.info(f"Evaluation started: iter={it}")
            figure = sampling(opt, it, val_dataloader, net, ema, scheduler, logger, writer)
            writer.add_figure(it, "log images", figure)
            net.train()

        th.save({
                    "net": net.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sched": sched.state_dict() if sched is not None else sched,
                }, opt.ckpt_path / "latest.pt")
        logger.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
    writer.close()

def load_from_ckpt(net, opt, logger):
    checkpoint = th.load(opt.ckpt_path, map_location="cpu")
    net.load_state_dict(checkpoint['net'])
    logger.info(f"[Net] Loaded network ckpt: {opt.ckpt_path}!")
    ema = ExponentialMovingAverage(net.parameters(), decay=opt.ema)
    ema.load_state_dict(checkpoint["ema"])
    logger.info(f"[Ema] Loaded ema ckpt: {opt.ckpt_path}!")
    
    net.to(opt.device)
    ema.to(opt.device)
    return net, ema

def save_gif(path_to_imgs, path_to_save, range_list):
    import imageio

    path_to_imgs = Path(path_to_imgs)
    images = [imageio.imread(str(path_to_imgs / f"{i}.png")) for i in range_list]
    imageio.mimsave(path_to_save, images)