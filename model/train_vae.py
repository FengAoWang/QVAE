#!/usr/bin/env python

# @Time    : 2025/4/13 15:01
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : train_vae.py

import os
import numpy as np
import anndata
import json
from sklearn.model_selection import train_test_split, ParameterGrid
from vae import VAE  # 导入新的VAE模型
from trainer import Trainer  # 复用原有Trainer
import torch
import scanpy as sc
import matplotlib.pyplot as plt

import os
import numpy as np
import anndata
import json
from sklearn.model_selection import train_test_split, ParameterGrid
from vae import VAE
from trainer import Trainer
import torch
import scanpy as sc
import matplotlib.pyplot as plt


def load_data(path: str) -> anndata.AnnData:
    """Load the data from a h5ad file."""
    return anndata.read_h5ad(path)


def split_data(adata: anndata.AnnData, test_size: float = 0.2, random_state: int = 512):
    """Split the AnnData object into training and testing sets."""
    indices = np.arange(adata.n_obs)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_data = adata[train_idx, :].copy()
    test_data = adata[test_idx, :].copy()
    return train_data, test_data


def get_representation(model, adata, device, batch_size=128):
    """Extract latent representations (z) from VAE."""
    model.eval()
    latent_reps = []
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    X = np.array(X)
    tensor_X = torch.tensor(X, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            _, _, _, z, _ = model(x)  # VAE返回的潜在变量为z
            latent_reps.append(z.cpu().numpy())
    latent_reps = np.concatenate(latent_reps, axis=0)
    return latent_reps


def run_training(training_config, adata):
    """Main training loop for VAE."""
    # Split data
    training_data, test_data = split_data(adata, test_size=0.2, random_state=training_config.get("random_state", 512))

    # Model initialization
    input_dim = adata.X.shape[1]
    model = VAE(
        input_dim=input_dim,
        hidden_dim=training_config["hidden_dim"],
        latent_dim=training_config["latent_dim"]
    )

    # Initialize trainer
    trainer = Trainer(model, training_config, training_data, test_data)
    trainer.train()

    # Test set evaluation
    test_result = trainer.val(test_data)
    test_result_file = os.path.join(training_config["output"], "test_result.json")
    with open(test_result_file, "w") as f:
        json.dump(test_result, f, indent=4)
    print("Test Results:", test_result)

    # Generate UMAP plot
    device = training_config.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    latent_test = get_representation(model, test_data, device)
    test_data.obsm['vae_reps'] = latent_test

    sc.pp.neighbors(test_data, n_neighbors=10, use_rep='vae_reps')
    sc.tl.umap(test_data)
    plt.figure(figsize=(4, 3))
    dataset_name = training_config.get("dataset_name", "RNA_filter")
    sc.pl.umap(test_data, color='cell_type', show=False)
    umap_file = os.path.join(training_config["output"], f'{dataset_name}_vae_cell_test.pdf')
    plt.savefig(umap_file, dpi=1000, bbox_inches='tight')
    print(f"UMAP plot saved as {umap_file}")

    return trainer


def main():
    # Base configuration (no RBM-related parameters)
    base_config = {
        "output": "./results_vae/",  # Dedicated output directory
        "batch_size": 128,
        "hidden_dim": 512,
        "latent_dim": 256,
        "lr": 1e-3,
        "num_epochs": 300,
        "save_log": True,
        "random_seed": 42,
        "random_state": 512,
        "dataset_name": "PBMC"
    }

    # Parameter grid search space (VAE-specific)
    param_grid = {
        "latent_dim": [128, 256, 512],
        "hidden_dim": [512],
        "batch_size": [128, 256, 512, 1024]
    }
    grid = list(ParameterGrid(param_grid))

    # Load data (update path to your dataset)
    adata = load_data(r'/mnt/zhangzheng_group/xuany-54/DVAE-RBM-RNA_seq/data/BMMC_RNA_filter.h5ad')

    # Grid search training
    for params in grid:
        config = base_config.copy()
        config.update(params)
        # Generate unique output folder name
        folder_name = f"latent-{config['latent_dim']}_hidden-{config['hidden_dim']}_bs-{config['batch_size']}"
        config["output"] = os.path.join(base_config["output"], folder_name)
        os.makedirs(config["output"], exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"Training VAE with params: {params}")
        print(f"Output folder: {config['output']}")
        print(f"{'=' * 50}\n")

        # Start training
        run_training(config, adata)


if __name__ == "__main__":
    main()
