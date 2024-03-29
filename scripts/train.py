from toysb.utils import train
from toysb.model import SBModel
from toysb.scheduler import Scheduler
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
from toysb.logger import Logger
from toysb.utils import create_symmetric_beta_schedule
from toysb.dataset import get_pair_dataset
import os

def create_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="experiment ID")
    parser.add_argument("--dataset1", type=str, help="name for initial dataset")
    parser.add_argument("--dataset2", type=str, help="name for terminal dataset")
    parser.add_argument("--n_samples", type=int, default=10**4, help="number of samples for each dataset")
    parser.add_argument("--path_to_save", type=Path, default="", help="path to save data")
    parser.add_argument("--log-dir", type=Path, default=".log", help="path to log std outputs and writer data")
    parser.add_argument("--ckpt_path", type=Path, default="", help="path to save checkpoints")
    parser.add_argument("--gpu", type=int, default=None, help="choose a particular device")
    parser.add_argument("--num_steps", type=int, default=1000, help="number of steps")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of epochs")
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
    parser.add_argument("--val_log", type=int, default=100, help="number of points to log from validation dataset")

    opt = parser.parse_args()

    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    return opt

def main(opt):
    logger = Logger(opt.log_dir)
    logger.info("toySB training")
    train_pair_dataset, dim = get_pair_dataset(opt.n_samples, opt.dataset1, opt.dataset2, logger, path_to_save=opt.path_to_save, regime = "train")
    val_pair_dataset, dim = get_pair_dataset(opt.val_log, opt.dataset1, opt.dataset2, logger, path_to_save=opt.path_to_save, regime = "val")
    net = SBModel(x_dim = dim)
    scheduler = Scheduler(create_symmetric_beta_schedule(n_timestep=opt.num_steps, linear_end=opt.beta_max / opt.num_steps), opt.device)
    train_dataloader = DataLoader(train_pair_dataset, opt.batch_size, shuffle = True)
    val_dataloader = DataLoader(val_pair_dataset, opt.val_log)
    train(opt, net, scheduler, train_dataloader, val_dataloader, logger)

    logger.info("Finish!")


if __name__ == '__main__':
    opt = create_arguments()
    main(opt)