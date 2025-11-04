import torch.utils.data as data
import torchvision as tv


def get_mnist():
    dataset = tv.datasets.MNIST("./data", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])
    return data.DataLoader(train), data.DataLoader(val)
