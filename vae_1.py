import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: x.view(-1) - 0.5)
])

def train():
    batch_size = 256
    # dimension=784: 28*28
    train_data = datasets.MNIST(
        "~/.pytorch/mnist_data",
        download=True,
        train=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super(VAE, self).__init__()


if __name__ == "__main__":
    train()
