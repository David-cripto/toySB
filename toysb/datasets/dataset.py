import numpy as np
import torch as th
import torchvision.transforms as transforms
from I2SB.corruption.superresolution import build_sr4x
from torch.utils.data import Dataset
from torchvision import datasets


def build_train_transform(size):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
        transforms.Resize(size) 
    ])


def build_test_transform(size):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1), # [0,1] --> [-1, 1]
        transforms.Resize(size)
    ])


def get_dataset(name_dataset, *args, **kwargs):
    if name_dataset == "mnist":
        dataset = datasets.MNIST(*args, **kwargs)
    elif name_dataset == "emnist":
        kwargs['split']='byclass'
        dataset = datasets.EMNIST(*args, **kwargs)
    elif name_dataset == "cifar10":
        dataset = datasets.CIFAR10(*args, **kwargs)
    else:
        raise AssertionError(f'{name_dataset} dataset is not supported')
    
    return dataset


def get_correspondance_index(name_dataset):
    if name_dataset == "emnist":
        return [10, 12, 15, 18, 22, 23, 24, 25, 28, 29] #TODO add more variety
    else:
        return list(range(10))


def get_pair_dataset(opt, logger, train: bool = True): #TODO Don't understand why to create a function that creates a class

    PAIRED_MODE = opt.dataset2 is not None # paired mode

    # transform = build_train_transform() if train else build_test_transform() # TODO don't want to rotate letters and digits
    transform = build_test_transform(opt.image_size)
    logger.info(f"[Dataset] Download {opt.dataset1} to {opt.root}")
    target_dataset = get_dataset(opt.dataset1, root=opt.root, train=train, download=True, transform=transform)

    start_dataset = None
    if PAIRED_MODE:
        logger.info(f"[Dataset] Download {opt.dataset2} to {opt.root}")
        start_dataset = get_dataset(opt.dataset2, root=opt.root, train=train, download=True, transform=transform)

    class PairDataset(Dataset):
        def __init__(self):
            self.names = [opt.dataset1, opt.dataset2]
            self.target_dataset = target_dataset
            if PAIRED_MODE:
                self.start_dataset = start_dataset
                self.target2start = {} # Strict correspondance between labels 
                self.indexes = [
                    get_correspondance_index(name) for name in [opt.dataset1, opt.dataset2]
                ]
                for idx_t, idx_s in zip(*self.indexes):
                    target_idxs = np.where(np.array(self.target_dataset.targets) == idx_t)[0].tolist()
                    start_idxs = np.where(np.array(self.start_dataset.targets) == idx_s)[0].tolist()
                    self.target2start.update(dict(zip(target_idxs, start_idxs)))

                self.raw_idx = np.array(list(self.target2start.keys()))
                np.random.seed(42)
                np.random.shuffle(self.raw_idx)
            else:
                self.corruption_func = build_sr4x(opt, logger, "bicubic", opt.image_size)

        def __len__(self):
            if PAIRED_MODE:
                return len(self.raw_idx)
            else:
                return len(self.target_dataset)
        
        def __getitem__(self, index):
            img, label = self.target_dataset[index]
            if PAIRED_MODE:
                target_idx = self.raw_idx[index]
                start_idx = self.target2start[target_idx]
                imgs = [
                    self.target_dataset[target_idx][0],
                    self.start_dataset[start_idx][0]
                    ]

                for i in range(2):
                    if imgs[i].shape[0] == 1:
                        if self.names[i] == 'emnist':
                            imgs[i] = th.transpose(imgs[i], 1, 2)
                        imgs[i] = th.stack([imgs[i]] * 3, dim=1).squeeze(0)
                return imgs
            else:
                if img.shape[0] == 1:
                    img = th.stack([img, img, img], dim = 1).squeeze(0)
                corrupt_image = self.corruption_func(img.unsqueeze(0)).squeeze(0)
                return img, corrupt_image

    return PairDataset()