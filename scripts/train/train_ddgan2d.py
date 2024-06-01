import numpy as np
import matplotlib.pyplot as plt

from project.toySB.toysb.models.ddgan2d import DDGAN2DDiscriminator, DDGAN2DGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
from functools import partial
from tqdm import tqdm

from toysb import train_ddgan, Scheduler, Logger
import argparse
from pathlib import Path
from toysb.datasets.dataset2d import get_pair_dataset
from toysb.utils_ddgan import create_symmetric_beta_schedule_ddgan
import os

def create_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="experiment ID")
    parser.add_argument("--dataset1", type=str, help="name for initial dataset")
    parser.add_argument("--dataset2", type=str, help="name for terminal dataset")
    parser.add_argument("--n_samples", type=int, default=10**4, help="number of samples for each dataset")
    parser.add_argument("--gpu", type=int, default=None, help="choose a particular device")
    parser.add_argument("--n_epoch", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr_d", type=float, default=1e-4, help="learning rate for discriminator")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="learning rate for generator")
    parser.add_argument("--beta_1", type=float, default=0.5, help="beta1 parameter for Adam")
    parser.add_argument("--beta_2", type=float, default=0.9, help="beta2 parameter for Adam")
    parser.add_argument("--lazy_reg", type=int, default=1, help="lazy r1 regularization for generator")
    parser.add_argument("--r1_gamma", type=float, default=1e-4, help="gamma parameter for r1 regularization of generator")
    parser.add_argument("--num_timesteps", type=int, default=4, help="number of function evaluations")
    parser.add_argument("--beta_min", type=float, default=1e-4, help="min beta param for diffusion")
    parser.add_argument("--beta_max", type=float, default=2e-4, help="max beta param for diffusion")
    parser.add_argument("--use_ema", action="store_true", help="usage of EMA for generator training")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--path_to_save", type=Path, default="", help="path to save data")
    parser.add_argument("--log-dir", type=Path, default=".log", help="path to log std outputs and writer data")
    parser.add_argument("--path_to_save", type=Path, default="", help="path to save data")

    parser.add_argument("--save_ckpt_every", type=int, default=5000, help="frequency for checkpoint saving")
    parser.add_argument("--save_content_every", type=int, default=5000, help="frequency for whole training setup saving")
    parser.add_argument("--visualize_every", type=int, default=1001, help="frequency of visualization")
    parser.add_argument("--print_every", type=int, default=100, help="frequency of logging")

    opt = parser.parse_args()

    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    opt.z_dim = 1
    opt.x_dim = 2
    opt.t_dim = 2
    opt.out_dim = 2

    return opt

def main(opt):
    logger = Logger(opt.log_dir)
    logger.info("toySB DDGAN training")

    train_pair_dataset, _ = get_pair_dataset(opt.n_samples, opt.dataset1, opt.dataset2, logger, path_to_save=opt.path_to_save, regime = "train")
    val_pair_dataset, _ = get_pair_dataset(opt.val_log, opt.dataset1, opt.dataset2, logger, path_to_save=opt.path_to_save, regime = "val")

    train_dataloader = DataLoader(train_pair_dataset, opt.batch_size, shuffle = True)
    valid_dataloader = DataLoader(val_pair_dataset, opt.batch_size)

    netG = DDGAN2DGenerator(
        x_dim = opt.x_dim,
        t_dim = opt.t_dim,
        z_dim = opt.z_dim,
        out_dim = opt.out_dim,
        n_t = opt.num_timesteps + 1,
        layers = [256, 256, 256],
    ).to(opt.device)

    netD = DDGAN2DDiscriminator(
        x_dim = opt.x_dim,
        t_dim = opt.t_dim,
        n_t = opt.num_timesteps + 1,
        layers = [256, 256, 256],
    ).to(opt.device)

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas = (opt.beta1, opt.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas = (opt.beta1, opt.beta2))

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, len(train_dataloader) * opt.n_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, len(train_dataloader) * opt.n_epoch, eta_min=1e-5)

    diff_scheduler = Scheduler(create_symmetric_beta_schedule_ddgan(n_timestep=opt.num_timesteps+1, linear_start=opt.beta_min, linear_end=opt.beta_max), device)

    train_ddgan(
        netG,
        netD,
        optimizerG,
        optimizerD,
        schedulerG,
        schedulerD,
        diff_scheduler,
        train_dataloader,
        **opt,
    )

    logger.info("Finish!")

if __name__ == '__main__':
    opt = create_arguments()
    main(opt)
