from toysb.utils import train
from toysb.model import SBModel
from toysb.scheduler import Scheduler
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
from toysb.logger import Logger
from toysb.utils import make_beta_schedule
from toysb.dataset import get_pair_dataset

def create_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset1", type=str, help="name for initial dataset")
    parser.add_argument("--dataset2", type=str, help="name for terminal dataset")
    parser.add_argument("--n_samples", type=int, default=10**4, help="number of samples for each dataset")
    parser.add_argument("--path_to_save", type=Path, default="", help="path to save data")
    parser.add_argument("--log-dir", type=Path, default=".log", help="path to log std outputs and writer data")
    parser.add_argument("--gpu", type=int, default=None, help="choose a particular device")
    parser.add_argument("--num_steps", type=int, default=1000, help="number of steps")
    parser.add_argument("--beta_max", type=float, default=0.3, help="max diffusion")
    parser.add_argument("--batch_size", type=int, default=128)

    opt = parser.parse_args()

    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'

    return opt

def main(opt):
    logger = Logger(opt.log_dir)
    logger.info("toySB training")
    pair_dataset, dim = get_pair_dataset(opt.n_samples, opt.dataset1, opt.dataset2, logger, path_to_save=opt.path_to_save)
    net = SBModel(dim)
    scheduler = Scheduler(make_beta_schedule(n_timestep=opt.num_steps, linear_end=opt.beta_max / opt.num_steps), opt.device)
    train_dataloader = DataLoader(pair_dataset, opt.batch_size, shuffle = True)
    train(opt, net, scheduler, train_dataloader)

    logger.info("Finish!")


if __name__ == '__main__':
    opt = create_arguments()
    main(opt)