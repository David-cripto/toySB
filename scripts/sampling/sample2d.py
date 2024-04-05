from toysb import Logger, SB2D, Scheduler, load_from_ckpt, sampling
from toysb.utils import create_symmetric_beta_schedule
from toysb.datasets.dataset2d import load_dataset
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import os

def create_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset", type=Path, help="path to terminal dataset")
    parser.add_argument("--path_to_save", type=Path, default="", help="path to save data")
    parser.add_argument("--log-dir", type=Path, default=".log", help="path to log std outputs and writer data")
    parser.add_argument("--ckpt_path", type=Path, default="", help="path to load checkpoints")
    parser.add_argument("--log_count", type=int, default=5, help="number of steps to log")
    parser.add_argument("--gpu", type=int, default=None, help="choose a particular device")
    parser.add_argument("--ot_ode", action="store_true", help="use OT-ODE model")
    parser.add_argument("--vel", action="store_true", help="draw velocity of motion")
    parser.add_argument("--ot_vel", action="store_true", help="draw ot_ode velocity of motion")
    parser.add_argument("--exp_int_vel", action="store_true", help="draw exponential integrator velocity of motion")
    parser.add_argument("--beta_max", type=float, default=0.3, help="max diffusion")
    parser.add_argument("--num_steps", type=int, default=1000, help="number of steps")
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--ab_order", type=int, default=0, help="order of polynom in Exponential Integrator")

    opt = parser.parse_args()

    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    os.makedirs(opt.log_dir, exist_ok=True)

    return opt

def main(opt):
    logger = Logger(opt.log_dir)
    logger.info("toySB sampling")
    val_dataset, dim = load_dataset(opt.path_to_dataset, logger)

    net = SB2D(x_dim = dim)
    scheduler = Scheduler(create_symmetric_beta_schedule(n_timestep=opt.num_steps, linear_end=opt.beta_max / opt.num_steps), opt.device)
    net, ema = load_from_ckpt(net, opt, logger)

    val_dataloader = DataLoader(val_dataset, len(val_dataset))
    
    sampling(opt, val_dataloader, net, ema, scheduler, opt.path_to_save)

    logger.info("Finish!")


if __name__ == '__main__':
    opt = create_arguments()
    main(opt)