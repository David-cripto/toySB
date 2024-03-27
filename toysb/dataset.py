from sklearn import datasets
import torch as th
from torch.utils.data import Dataset
from pathlib import Path

PathLike = Path | str

def get_dataset(n_samples, name_dataset):
    if name_dataset == "scurve":
        X, y  = datasets.make_s_curve(n_samples=n_samples, noise=0.0, random_state=None)
        init_sample = th.tensor(X)[:,[0,2]]
        scaling_factor = 7
        init_sample = (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor

    elif name_dataset == "swiss":
        X, y  = datasets.make_swiss_roll(n_samples=n_samples, noise=0.0, random_state=None)
        init_sample = th.tensor(X)[:,[0,2]]
        scaling_factor = 7
        init_sample = (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor

    init_sample = init_sample.float()

    return init_sample

def get_pair_dataset(n_samples: int, dataset1: str, dataset2: str, transforms = None, path_to_save: PathLike = None):
    samples1 = get_dataset(n_samples, dataset1)
    samples2 = get_dataset(n_samples, dataset2)

    if path_to_save is not None:
        path_to_save = Path(path_to_save)
        th.save(samples1, path_to_save / "samples1.th")
        th.save(samples2, path_to_save / "samples2.th")

    class PairDataset(Dataset):
        def __init__(self):
            self.samples1 = transforms(samples1) if transforms is not None else samples1
            self.samples2 = transforms(samples2) if transforms is not None else samples2

        def __len__(self):
            return len(self.samples1)
        
        def __getitem__(self, index):
            return self.samples1[index], self.samples2[index]
    
    return PairDataset()

