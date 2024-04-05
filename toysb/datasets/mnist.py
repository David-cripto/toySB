from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def build_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) 
    ])


def build_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # [0,1] --> [-1, 1]
    ])


def get_dataset(name_dataset, *args, **kwargs):
    if name_dataset == "mnist":
        dataset = datasets.MNIST(*args, **kwargs)
    elif name_dataset == "emnist":
        dataset = datasets.EMNIST(*args, **kwargs)
    return dataset


def get_pair_dataset(opt, logger, corruption_func, train: bool = True):
    transform = build_train_transform() if train else build_test_transform()
    logger.info(f"[Dataset] Download {opt.dataset} to {opt.root}")
    hr_dataset = get_dataset(opt.dataset, root = opt.root, train = train, download = True, transform = transform)

    class PairDataset(Dataset):
        def __init__(self):
            self.hr_dataset = hr_dataset

        def __len__(self):
            return len(self.hr_dataset)
        
        def __getitem__(self, index):
            img, label = self.hr_dataset[index]
            return img, corruption_func(img)

    return PairDataset()