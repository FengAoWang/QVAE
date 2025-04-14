import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform


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
    def __init__(self, latent_dim, use_ising_sampling=False):
        super(RBM, self).__init__()
        self.h = nn.Parameter(torch.zeros(latent_dim))
        self.W = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.001)  # 对称权重
        self.latent_dim = latent_dim
        self.use_ising_sampling = use_ising_sampling

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

    def torch_map_clip(self, old, coup, bias, a, b, noi):
        noise = torch.randn(old.shape, device=old.device) * noi
        out = a * old + b * torch.matmul(coup, old) + 0.40 * b * bias + noise
        return torch.clamp(out, min=-0.4, max=0.4)

    def ising_sampling(self, num_samples, n_hidden):
        d = n_hidden + n_hidden  # 总节点数
        ad = torch.zeros((d, d), device=self.h.device)
        for i in range(n_hidden):
            for j in range(n_hidden):
                value = 0.25 * self.W[i, j]
                ad[i, n_hidden + j] = value
                ad[n_hidden + j, i] = value

        bias = torch.zeros(d, device=self.h.device)
        for i in range(n_hidden):
            bias[i] = 0.5 * self.h[i] + 0.25 * torch.sum(self.W[i, :])
        for j in range(n_hidden):
            bias[n_hidden + j] = 0.5 * self.h[j] + 0.25 * torch.sum(self.W[:, j])

        a = 0.9
        b = 0.1
        noise_strength = 0.12

        chain = torch.zeros(d, device=self.h.device)
        samples = []
        for _ in range(num_samples):
            chain = self.torch_map_clip(chain, ad, bias, a, b, noise_strength)
            samples.append(chain.clone())
        result = torch.stack(samples, dim=0)
        z = 0.5 * (torch.sign(result) + 1)
        return z[:, :n_hidden]

    def sample_hiddens(self, num_samples):
        if self.use_ising_sampling:
            return self.ising_sampling(num_samples, self.latent_dim)
        else:
            return self.gibbs_sampling(num_samples)

    def compute_gradients(self, z_positive, num_negative_samples=64, gibbs_steps=30):
        z_positive = z_positive.float()
        positive_h_grad = z_positive.mean(dim=0)  # ∂E/∂h = z_l
        positive_w_grad = torch.einsum('bi,bj->ij', z_positive, z_positive) / z_positive.size(0)  # ∂E/∂W = z_l z_m

        if self.use_ising_sampling:
            z_negative = self.ising_sampling(num_negative_samples, self.latent_dim)
        else:
            z_negative = self.gibbs_sampling(num_negative_samples, steps=gibbs_steps)
        negative_h_grad = z_negative.mean(dim=0)
        negative_w_grad = torch.einsum('bi,bj->ij', z_negative, z_negative) / z_negative.size(0)
        h_grad = positive_h_grad - negative_h_grad
        w_grad = positive_w_grad - negative_w_grad
        w_grad = (w_grad + w_grad.T) / 2
        return {'h': h_grad, 'W': w_grad}


class DVAE_RBM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=1.0, alpha=0.001, use_ising_sampling=False):
        super(DVAE_RBM, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.rbm = RBM(latent_dim, use_ising_sampling)
        self.beta = beta
        self.alpha = alpha
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

    def kl_divergence(self, z, q):
        q = torch.clamp(q, min=1e-7, max=1 - 1e-7)
        log_q = z * torch.log(q) + (1 - z) * torch.log(1 - q)
        entropy = -log_q.sum(dim=-1)
        energy_pos = self.rbm.energy(z)
        z_negative = self.rbm.sample_hiddens(z.size(0))
        energy_neg = self.rbm.energy(z_negative)
        logZ = energy_neg.mean()
        kl = (energy_pos - entropy + logZ).mean()
        return kl

    def forward(self, x):
        q_logits = self.encoder(x)
        rho = Uniform(0, 1).sample(q_logits.shape).to(x.device)
        zeta, z, q = self.reparameterize(q_logits, rho)
        x_recon = self.decoder(zeta)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl_loss = self.kl_divergence(z, q)
        elbo = -recon_loss - self.alpha * kl_loss
        return elbo, recon_loss, kl_loss, z, zeta