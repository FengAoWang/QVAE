import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from torch.utils.data import Dataset


class scDataset(Dataset):
    def __init__(self, anndata_info):
        self.rna_tensor = torch.tensor(anndata_info, dtype=torch.float32)

    def __len__(self):
        return self.rna_tensor.shape[0]

    def __getitem__(self, idx):
        return self.rna_tensor[idx, :]


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        q_logits = self.fc2(h)
        return q_logits


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, zeta):
        h = F.relu(self.bn1(self.fc1(zeta)))
        x_recon = self.fc2(h)
        return x_recon


class RBM(nn.Module):
    def __init__(self, latent_dim):
        super(RBM, self).__init__()
        self.h = nn.Parameter(torch.zeros(latent_dim))
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.001)  # 对称权重
        self.latent_dim = latent_dim

    def energy(self, z):
        z = z.float()
        h_term = torch.sum(z * self.h, dim=-1)
        w_term = torch.sum((z @ self.W) * z, dim=-1)  # 注意对称性
        return h_term + w_term

    def gibbs_sampling(self, num_samples, steps=30):
        z = torch.randint(0, 2, (num_samples, self.latent_dim), dtype=torch.float).to(self.h.device)
        for _ in range(steps):
            probs = torch.sigmoid(self.h + z @ self.W)
            z = (torch.rand_like(z) < probs).float()
        return z

    def compute_gradients(self, z_positive, num_negative_samples=64, gibbs_steps=30):
        """计算正相和负相的梯度"""
        # 正相：E_q[∂E/∂θ]
        z_positive = z_positive.float()
        positive_h_grad = z_positive.mean(dim=0)  # ∂E/∂h = z_l
        positive_w_grad = torch.einsum('bi,bj->ij', z_positive, z_positive) / z_positive.size(0)  # ∂E/∂W = z_l z_m

        # 负相：E_p[∂E/∂θ]
        z_negative = self.gibbs_sampling(num_negative_samples, steps=gibbs_steps)
        negative_h_grad = z_negative.mean(dim=0)
        negative_w_grad = torch.einsum('bi,bj->ij', z_negative, z_negative) / z_negative.size(0)

        # 总梯度
        h_grad = positive_h_grad - negative_h_grad
        w_grad = positive_w_grad - negative_w_grad
        # 对称化W的梯度（因为RBM假设W对称）
        w_grad = (w_grad + w_grad.T) / 2
        return {'h': h_grad, 'W': w_grad}


class DVAE_RBM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=1.0):
        super(DVAE_RBM, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.rbm = RBM(latent_dim)
        self.beta = beta
        self.latent_dim = latent_dim

    def reparameterize(self, q_logits, rho):
        q = torch.sigmoid(q_logits)
        zeta = torch.zeros_like(rho)
        mask = rho > (1 - q)
        beta_tensor = torch.tensor(self.beta, dtype=torch.float32, device=rho.device)
        exp_beta_minus_1 = torch.exp(beta_tensor) - 1
        zeta[mask] = (1 / beta_tensor) * torch.log(
            (torch.clamp(rho[mask] - (1 - q[mask]), min=0) / q[mask]) * exp_beta_minus_1 + 1
        )
        z = (zeta > 0).float()
        return zeta, z, q

    # def kl_divergence(self, z, q):
    #     # 熵项 H(q_phi)
    #     q = torch.clamp(q, min=1e-7, max=1 - 1e-7)
    #     entropy = -(q * torch.log(q) + (1 - q) * torch.log(1 - q)).sum(dim=-1)
    #     # 交叉熵项 H(q_phi, p_theta) = E[E_theta] + log Z_theta
    #     energy = self.rbm.energy(z).mean()
    #     # 负相通过采样近似（这里仅用于监控，不直接影响梯度）
    #     z_samples = self.rbm.gibbs_sampling(z.size(0))
    #     energy_samples = self.rbm.energy(z_samples).mean()
    #     cross_entropy = energy  # log Z_theta 被正负相差分抵消
    #     return (entropy - (-cross_entropy)).mean()

    def kl_divergence(self, z, q):
        q = torch.clamp(q, min=1e-7, max=1 - 1e-7)
        log_q = z * torch.log(q) + (1 - z) * torch.log(1 - q)
        entropy = -log_q.sum(dim=-1)
        energy_pos = self.rbm.energy(z)
        z_negative = self.rbm.gibbs_sampling(z.size(0))
        energy_neg = self.rbm.energy(z_negative)
        # 用负相能量的均值作为 logZ 的近似
        logZ = energy_neg.mean()
        # print(energy_pos)
        # print(entropy)
        # print(logZ)
        kl = (energy_pos - entropy + logZ).mean()
        return kl

    def forward(self, x):
        q_logits = self.encoder(x)
        rho = Uniform(0, 1).sample(q_logits.shape).to(x.device)
        zeta, z, q = self.reparameterize(q_logits, rho)
        x_recon = self.decoder(zeta)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl_loss = self.kl_divergence(z, q)
        elbo = -recon_loss - 0.001 * kl_loss
        return elbo, recon_loss, kl_loss, z, zeta



