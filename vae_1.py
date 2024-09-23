import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import torch.nn.functional as F

from dataclasses import dataclass
from tqdm.auto import tqdm


@dataclass
class VAEOutput:
    z_dist: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor

    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, 2 * latent_dim)
        )

        self.softplus = nn.Softplus()

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim //2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x, eps: float=1e-8):
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist):
        return dist.rsample()

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, compute_loss: bool=True):
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None
            )

        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(
                z.shape[-1],
                device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1)
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
        loss_recon = F.binary_cross_entropy(recon_x, x + 0.5, reduction="none").sum(-1).mean()
        loss = loss_recon + loss_kl

        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl
        )


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
    # hyperparameters
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 1e-2
    num_epochs = 50
    latent_dim = 2
    hidden_dim = 512
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
    model = VAE(
        input_dim=784,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} / {num_epochs}")
        for batch_idx, (data, _) in enumerate((train_loader)):
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = output.loss
            loss.backward()

            if batch_idx % 50 == 0:
                print(f"Epoch: {epoch + 1} Batch: {batch_idx + 1} Loss: {loss} Loss_kl: {output.loss_kl} Loss_recon: {output.loss_recon}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()




if __name__ == "__main__":
    train()
