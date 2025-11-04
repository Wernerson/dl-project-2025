import torch.utils.data as data
import torchvision as tv


def mnist(data_dir, train_val_split):
    dataset = tv.datasets.MNIST(data_dir, download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, train_val_split)
    return data.DataLoader(train), data.DataLoader(val)
