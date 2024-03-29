from sklearn import datasets
import torch as th
from torch.utils.data import Dataset
from pathlib import Path
from .logger import Logger

PathLike = Path | str

def get_dataset(n_samples, name_dataset):
    if name_dataset == "scurve":
        X, y  = datasets.make_s_curve(n_samples=n_samples, noise=0.0, random_state=None)
        init_sample = th.tensor(X)[:,[0,2]]
        scaling_factor = 7
        init_sample = (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor
        dim = init_sample.shape[1]

    elif name_dataset == "swiss":
        X, y  = datasets.make_swiss_roll(n_samples=n_samples, noise=0.0, random_state=None)
        init_sample = th.tensor(X)[:,[0,2]]
        scaling_factor = 7
        init_sample = (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor
        dim = init_sample.shape[1]

    init_sample = init_sample.float()

    return init_sample, dim

def get_pair_dataset(
        n_samples: int, dataset1: str, dataset2: str, 
        logger: Logger, transforms = None, 
        path_to_save: PathLike = None, regime: str = "train"
        ):
    samples1, dim = get_dataset(n_samples, dataset1)
    samples2, dim = get_dataset(n_samples, dataset2)

    if path_to_save is not None:
        path_to_save = Path(path_to_save) / regime
        path_to_save.mkdir(exist_ok=True)
        path1, path2 = str(path_to_save / "samples1.th"), str(path_to_save / "samples2.th")
        th.save(samples1, path1)
        logger.info(f"[Dataset] Save {dataset1} to {path1}")
        th.save(samples2, path2)
        logger.info(f"[Dataset] Save {dataset2} to {path2}")

    class PairDataset(Dataset):
        def __init__(self):
            self.samples1 = transforms(samples1) if transforms is not None else samples1
            self.samples2 = transforms(samples2) if transforms is not None else samples2

        def __len__(self):
            return len(self.samples1)
        
        def __getitem__(self, index):
            return self.samples1[index], self.samples2[index]
    logger.info(f"[Dataset] Built {dataset1} and {dataset2} datasets, size={len(samples1)}!")
    return PairDataset(), dim

