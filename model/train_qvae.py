#!/usr/bin/env python

# @Time    : 2025/4/13 15:33
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : train.copy.py

import os
import numpy as np
import anndata
import json
from sklearn.model_selection import train_test_split, ParameterGrid
from qvae_rbm import DVAE_RBM
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
            _, _, _, _, zeta = model(x)
            latent_reps.append(zeta.cpu().numpy())
    latent_reps = np.concatenate(latent_reps, axis=0)
    return latent_reps


def run_training(training_config, adata):
    training_data, test_data = split_data(adata, test_size=0.2, random_state=training_config.get("random_state", 512))
    input_dim = adata.X.shape[1]

    # 根据配置创建模型
    model = DVAE_RBM(
        input_dim=input_dim,
        hidden_dim=training_config["hidden_dim"],
        latent_dim=training_config["latent_dim"],
        beta=training_config["beta"],
        alpha=training_config["alpha"],
        use_ising_sampling=training_config.get("sampling_method", "gibbs") == "ising"
    )

    # 根据配置中的输出文件夹名称保存各fold及最终聚合结果
    trainer = Trainer(model, training_config, training_data, test_data)
    trainer.train()

    # 对测试集进行评估
    test_result = trainer.val(test_data)
    test_result_file = os.path.join(training_config["output"], "test_result.json")
    with open(test_result_file, "w") as f:
        json.dump(test_result, f, indent=4)
    print("Test Results:", test_result)

    # 为测试数据生成隐变量表示并绘制UMAP图
    device = training_config.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    latent_test = get_representation(model, test_data, device)
    test_data.obsm['qvae_reps'] = latent_test

    sc.pp.neighbors(test_data, n_neighbors=10, use_rep='qvae_reps')
    sc.tl.umap(test_data)
    plt.figure(figsize=(4, 3))
    dataset_name = training_config.get("dataset_name", "RNA_filter")
    sc.pl.umap(test_data, color='cell_type', show=False)
    umap_file = os.path.join(training_config["output"], f'{dataset_name}_qvae_cell_test.pdf')
    plt.savefig(umap_file, dpi=1000, bbox_inches='tight')
    print(f"UMAP plot saved as {umap_file}")

    return trainer


def main():
    base_config = {
        "output": "./results/",
        "batch_size": 128,
        "hidden_dim": 512,
        "latent_dim": 256,
        "beta": 0.5,
        "alpha": 0.001,
        "lr": 1e-3,
        "rbm_lr": 1e-4,
        "num_epochs": 300,
        "num_negative_samples": 64,
        "save_log": True,
        "random_seed": 42,
        "random_state": 512,
        "dataset_name": "BMMC"
    }

    # 定义网格搜索参数空间，新增 sampling_method
    param_grid = {
        "sampling_method": ["gibbs"],  # 新增采样方式参数
        "beta": [0.5, 1.0, 0.75, 0.25],
        "alpha": [0.001, 0.01, 1, 0],
        "latent_dim": [256, 128, 512],
        "batch_size": [128, 256, 512, 1024],
    }
    grid = list(ParameterGrid(param_grid))

    adata = load_data(r'/mnt/zhangzheng_group/xuany-54/DVAE-RBM-RNA_seq/data/BMMC_RNA_filter.h5ad')

    for params in grid:
        config = base_config.copy()
        config.update(params)
        # 生成唯一输出文件夹名称，包含采样方式

        folder_name = (
            f"beta-{config['beta']}_alpha-{config['alpha']}_latent-{config['latent_dim']}_"
            f"hidden-{config['hidden_dim']}_sampling-{config['sampling_method']}_batch-{config['batch_size']}"
        )
        config["output"] = os.path.join(base_config["output"], folder_name)
        os.makedirs(config["output"], exist_ok=True)

        print(f"Running training for configuration: {params}, output folder: {config['output']}")
        trainer = run_training(config, adata)


if __name__ == "__main__":
    main()
