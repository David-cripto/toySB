from toysb import Logger, Unet, Scheduler, load_from_ckpt, sampling, get_model
from toysb.utils import create_symmetric_beta_schedule, sample_and_save
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import os
from toysb.datasets.dataset import get_pair_dataset
from I2SB.corruption.superresolution import build_sr4x

def create_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset1", type=str, help="name of the dataset")
    parser.add_argument("--dataset2", type=str, help="name of the dataset")
    parser.add_argument("--root", type=Path, default="", help="path to dataset")
    parser.add_argument("--path_to_save", type=Path, default="", help="path to save data")
    parser.add_argument("--log-dir", type=Path, default=".log", help="path to log std outputs and writer data")
    parser.add_argument("--ckpt_path", type=Path, default="", help="path to load checkpoints")
    parser.add_argument("--log_count", type=int, default=5, help="number of steps to log")
    parser.add_argument("--gpu", type=int, default=None, help="choose a particular device")
    parser.add_argument("--ddpm", action="store_true", help="use DDPM")
    parser.add_argument("--ot_ode", action="store_true", help="use OT-ODE model")
    parser.add_argument("--exp_int", action="store_true", help="use Exponential Integrator")
    parser.add_argument("--beta_max", type=float, default=0.3, help="max diffusion")
    parser.add_argument("--num_steps", type=int, default=1000, help="number of steps")
    parser.add_argument("--ema", type=float, default=0.99)
    parser.add_argument("--ab_order", type=int, default=0, help="order of polynom in Exponential Integrator")
    parser.add_argument("--c_in", type=int, default=3, help="in channels")
    parser.add_argument("--val_log", type=int, default=10, help="number of points to log from validation dataset")
    parser.add_argument("--image_size", type=int, default=28, help="image size")
    parser.add_argument("--nfe", type=int, default=1000, help="number of function evaluations")
    parser.add_argument("--save_raw_data", action="store_true", help="use OT-ODE model")
    
    opt = parser.parse_args()

    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    os.makedirs(opt.log_dir, exist_ok=True)

    opt.vel = False
    opt.exp_int_vel = False

    return opt

def main(opt):
    assert sum([opt.ddpm, opt.ot_ode, opt.exp_int]) == 1, "Should be only one regime of sampling during sampling"
    logger = Logger(opt.log_dir)
    logger.info("toySB sampling")
    val_pair_dataset = get_pair_dataset(opt, logger, train=False)

    net = get_model(image_size = opt.image_size, in_channels = opt.c_in, num_channels = 64, num_res_blocks = 5)
    scheduler = Scheduler(create_symmetric_beta_schedule(n_timestep=opt.num_steps, linear_end=opt.beta_max / opt.num_steps), opt.device)
    net, ema = load_from_ckpt(net, opt, logger)

    val_dataloader = DataLoader(val_pair_dataset, opt.val_log)

    experiment_name = Path(opt.ckpt_path).parts[-2]
    path_to_save = Path(opt.log_dir) / experiment_name / Path(opt.ckpt_path).stem / 'images'
    path_to_save.mkdir(parents=True, exist_ok=True)
    
    sampling(opt, val_dataloader, net, ema, scheduler, path_to_save)

    logger.info("Finish!")


if __name__ == '__main__':
    opt = create_arguments()
    main(opt)