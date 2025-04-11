import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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
    def __init__(self,
                 input_dim,
                 hidden_dim=512,
                 latent_dim=256,
                 beta=0.5,
                 beta_kl=0.001,
                 use_batch_norm=True,
                 use_layer_norm=True,
                 device=torch.device('cpu')):
        super(DVAE_RBM, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.rbm = RBM(latent_dim)
        self.beta = beta
        self.latent_dim = latent_dim
        self.device = device
        self.beta_kl = beta_kl

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
        elbo = -recon_loss - self.beta_kl * kl_loss
        return elbo, recon_loss, kl_loss, z, zeta

    def get_representation(self,
                           adata,
                           batch_size=128,):
        self.eval()
        latent_reps = []
        adata_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        sc_dataset = scDataset(adata_array)
        with torch.no_grad():
            for x in DataLoader(sc_dataset, batch_size=batch_size, shuffle=False, num_workers=4):
                x = x.to(self.device)
                _, _, _, _, zeta = self(x)
                latent_reps.append(zeta.cpu().numpy())
        reps = np.concatenate(latent_reps, axis=0)
        print(f"Latent representation shape: {reps.shape}")
        return reps

    def fit(self,
            adata,
            val_percentage=0.1,
            batch_size=128,
            epochs=100,
            beta=0.5,
            lr=1e-3,
            rbm_lr=1e-4,
            early_stopping=True,
            early_stopping_patience=15,):

        if early_stopping:
            train_indices, val_indices = train_test_split(np.arange(adata.shape[0]), test_size=val_percentage, random_state=0)

            adata_train = adata[train_indices, :].copy()
            adata_val = adata[val_indices, :].copy()

            adata_train_array = adata_train.X.toarray() if hasattr(adata_train.X, 'toarray') else adata_train.X
            adata_val_array = adata_val.X.toarray() if hasattr(adata_val.X, 'toarray') else adata_val.X

            sc_val_dataset = scDataset(adata_val_array)
            val_dataloader = DataLoader(sc_val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            adata_train_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        sc_train_dataset = scDataset(adata_train_array)
        train_dataloader = DataLoader(sc_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        rbm_optimizer = torch.optim.Adam(self.rbm.parameters(), lr=rbm_lr)

        # Early stopping variables
        best_val_elbo = float('-inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.train()
            total_elbo, total_recon, total_kl = 0, 0, 0
            for batch_idx, x in enumerate(train_dataloader):
                x = x.to(self.device)
                optimizer.zero_grad()
                rbm_optimizer.zero_grad()

                elbo, recon_loss, kl_loss, z, zeta = self(x)
                loss = -elbo
                loss.backward()

                # 手动计算RBM的梯度
                rbm_grads = self.rbm.compute_gradients(z.detach())  # z.detach()避免重复求导
                with torch.no_grad():
                    self.rbm.h.grad = rbm_grads['h']
                    self.rbm.W.grad = rbm_grads['W']

                optimizer.step()
                rbm_optimizer.step()

                total_elbo += elbo.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            avg_elbo = total_elbo / len(train_dataloader)
            avg_recon = total_recon / len(train_dataloader)
            avg_kl = total_kl / len(train_dataloader)
            print(f"Epoch [{epoch}/{epochs}], ELBO: {avg_elbo:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

            if early_stopping:
                self.eval()
                val_total_elbo, val_total_recon, val_total_kl = 0, 0, 0
                for batch_idx, x in enumerate(val_dataloader):
                    x = x.to(self.device)
                    with torch.no_grad():
                        elbo, recon_loss, kl_loss, z, zeta = self(x)

                    val_total_elbo += elbo.item()
                    val_total_recon += recon_loss.item()
                    val_total_kl += kl_loss.item()

                avg_val_elbo = val_total_elbo / len(val_dataloader)
                avg_recon = val_total_recon / len(val_dataloader)
                avg_kl = val_total_kl / len(val_dataloader)

                # Early stopping logic
                if avg_val_elbo > best_val_elbo:
                    best_val_elbo = avg_val_elbo
                    patience_counter = 0
                    print("Best model updated")
                else:
                    patience_counter += 1
                    print(f"Patience counter: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break


class VAEEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 latent_dim):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        # For VAE, we need two outputs: mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAE(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=512,
                 latent_dim=256,
                 beta_kl=0.001,
                 device=torch.device('cpu')):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.beta_kl = beta_kl
        self.latent_dim = latent_dim
        self.device = device

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def kl_divergence(self, mu, logvar):
        # Analytical KL divergence for Gaussian prior N(0,1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar)
        # ELBO (Evidence Lower Bound)
        elbo = -recon_loss - 0.001 * kl_loss

        return elbo, recon_loss, kl_loss, z

    def fit(self,
            adata,
            val_percentage=0.1,
            batch_size=128,
            epochs=100,
            lr=1e-3,
            early_stopping=True,
            early_stopping_patience=10,
            ):
        if early_stopping:
            train_indices, val_indices = train_test_split(np.arange(adata.shape[0]), test_size=val_percentage, random_state=0)

            adata_train = adata[train_indices, :].copy()
            adata_val = adata[val_indices, :].copy()

            adata_train_array = adata_train.X.toarray() if hasattr(adata_train.X, 'toarray') else adata_train.X
            adata_val_array = adata_val.X.toarray() if hasattr(adata_val.X, 'toarray') else adata_val.X

            sc_val_dataset = scDataset(adata_val_array)
            val_dataloader = DataLoader(sc_val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            adata_train_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

        sc_train_dataset = scDataset(adata_train_array)
        train_dataloader = DataLoader(sc_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr
        )

        # Early stopping variables
        best_val_elbo = float('-inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.train()
            total_elbo, total_recon, total_kl = 0, 0, 0
            for batch_idx, x in enumerate(train_dataloader):
                x = x.to(self.device)
                optimizer.zero_grad()

                elbo, recon_loss, kl_loss, z = self(x)
                loss = -elbo
                loss.backward()
                optimizer.step()

                total_elbo += elbo.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            avg_elbo = total_elbo / len(train_dataloader)
            avg_recon = total_recon / len(train_dataloader)
            avg_kl = total_kl / len(train_dataloader)
            print(f"Epoch [{epoch}/{epochs}], ELBO: {avg_elbo:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

            if early_stopping:
                self.eval()
                val_total_elbo, val_total_recon, val_total_kl = 0, 0, 0
                for batch_idx, x in enumerate(val_dataloader):
                    x = x.to(self.device)
                    with torch.no_grad():
                        elbo, recon_loss, kl_loss, z = self(x)

                    val_total_elbo += elbo.item()
                    val_total_recon += recon_loss.item()
                    val_total_kl += kl_loss.item()

                avg_val_elbo = val_total_elbo / len(val_dataloader)
                avg_recon = val_total_recon / len(val_dataloader)
                avg_kl = val_total_kl / len(val_dataloader)

                # Early stopping logic
                if avg_val_elbo > best_val_elbo:
                    best_val_elbo = avg_val_elbo
                    patience_counter = 0
                    print("Best model updated")
                else:
                    patience_counter += 1
                    print(f"Patience counter: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

    def get_representation(self,
                           adata,
                           batch_size=128,):
        self.eval()
        latent_reps = []
        adata_array = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        sc_dataset = scDataset(adata_array)
        with torch.no_grad():
            for x in DataLoader(sc_dataset, batch_size=batch_size, shuffle=False, num_workers=4):
                x = x.to(self.device)
                _, _, _, z = self(x)
                latent_reps.append(z.cpu().numpy())
        reps = np.concatenate(latent_reps, axis=0)
        print(f"Latent representation shape: {reps.shape}")
        return reps

