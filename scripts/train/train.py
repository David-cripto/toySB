from toysb import train, Scheduler, Logger, Unet, get_model
from toysb.datasets.dataset import get_pair_dataset
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
from toysb.utils import create_symmetric_beta_schedule
import os
from datetime import datetime
from I2SB.corruption.superresolution import build_sr4x

def create_arguments():
    now = datetime.now().strftime("%y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=now, help="experiment ID")
    parser.add_argument("--dataset1", type=str, help="name of the target dataset")
    parser.add_argument("--dataset2", type=str, help="name of the initial dataset", default=None)
    parser.add_argument("--root", type=Path, default="", help="path to save data")
    parser.add_argument("--log-dir", type=Path, default=".log", help="path to log std outputs and writer data")
    parser.add_argument("--ckpt_path", type=Path, default="", help="path to save checkpoints")
    parser.add_argument("--ckpt_every", type=int, default=0, help="period of checkpointing; 0 - save only last")
    parser.add_argument("--gpu", type=int, default=None, help="choose a particular device")
    parser.add_argument("--num_steps", type=int, default=1000, help="number of steps")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of epochs")
    parser.add_argument("--image_size", type=int, default=28, help="image size")
    parser.add_argument("--c_in", type=int, default=3, help="in channels")
    parser.add_argument("--beta_max", type=float, default=0.3, help="max diffusion")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--l2-norm", type=float, default=0.0)
    parser.add_argument("--lr-step", type=int, default=1000, help="learning rate decay step size")
    parser.add_argument("--lr-gamma", type=float, default=0.99, help="learning rate decay ratio")
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--t0", type=float, default=1e-4, help="start time in network parametrization")
    parser.add_argument("--T", type=float, default=1., help="end time in network parametrization")
    parser.add_argument("--log_count", type=int, default=5, help="number of steps to log")
    parser.add_argument("--val_log", type=int, default=10, help="number of points to log from validation dataset")
    parser.add_argument("--ddpm", action="store_true", help="use DDPM")
    parser.add_argument("--ot_ode", action="store_true", help="use OT-ODE model")
    parser.add_argument("--exp_int", action="store_true", help="use Exponential Integrator")
    parser.add_argument("--ab_order", type=int, default=0, help="order of polynom in Exponential Integrator")
    parser.add_argument("--nfe", type=int, default=1000, help="number of function evaluations")
    parser.add_argument("--verbose", action="store_true", help="verbosity level (bool)")

    opt = parser.parse_args()

    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    os.makedirs(opt.log_dir, exist_ok=True)
    (Path(opt.ckpt_path) / opt.name).mkdir(parents=True, exist_ok=True)

    opt.vel = False
    opt.exp_int_vel = False

    return opt

def main(opt):
    assert sum([opt.ddpm, opt.ot_ode]) == 1, "Should be only one regime of sampling during training"
    logger = Logger(opt.log_dir)
    logger.info("toySB training")
    train_pair_dataset = get_pair_dataset(opt, logger, train=True)
    val_pair_dataset = get_pair_dataset(opt, logger, train=False)
    net = get_model(image_size = opt.image_size, in_channels = opt.c_in, num_channels = 64, num_res_blocks = 5)
    scheduler = Scheduler(create_symmetric_beta_schedule(n_timestep=opt.num_steps, linear_end=opt.beta_max / opt.num_steps), opt.device)
    train_dataloader = DataLoader(train_pair_dataset, opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_pair_dataset, opt.val_log)
    train(opt, net, scheduler, train_dataloader, val_dataloader, logger)

    logger.info("Finish!")


if __name__ == '__main__':
    opt = create_arguments()
    main(opt)