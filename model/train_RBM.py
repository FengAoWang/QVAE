import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from model.DVAE_RBM import scDataset, DVAE_RBM
import torch
from utils import load_fold_indices, split_data, compute_clusters_performance, multiprocessing_train_fold, worker_function
import random
import logging


def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')


def train_fold(adata, dataset_name, fold_id, device_id=6):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_idx, test_idx = load_fold_indices(dataset_name, fold_num=fold_id)

    gex_adata_train = adata[train_idx, :].copy()
    gex_adata_test = adata[test_idx, :].copy()

    print(f"Data shape: {gex_adata_train.X.shape}")

    input_dim = gex_data.X.shape[1]
    model = DVAE_RBM(input_dim, device=device).to(device)

    model.fit(gex_adata_train, epochs=100)

    # Add latent representations to AnnData
    gex_adata_test.obsm['qvae_reps'] = model.get_representation(gex_adata_test)
    latent_test = gex_adata_test.obsm['qvae_reps']

    # UMAP and clustering
    sc.pp.neighbors(gex_adata_test, n_neighbors=10, use_rep='qvae_reps')
    sc.tl.umap(gex_adata_test)
    plt.figure(figsize=(4, 3))
    sc.pl.umap(gex_adata_test, color='cell_type', show=False)
    plt.savefig(f'{dataset_name}_qvae_cell_{fold_id}.pdf', dpi=1000, bbox_inches='tight')

    sc.pp.neighbors(gex_adata_test, n_neighbors=10, use_rep='qvae_reps', random_state=42)
    sc.tl.leiden(gex_adata_test, random_state=42)

    # Clustering metrics
    if 'cell_type' in gex_adata_test.obs:
        ARI, AMI, NMI, HOM, FMI = compute_clusters_performance(gex_adata_test, 'cell_type')
        print(f'ARI: {ARI:.4f}, AMI: {AMI:.4f}, NMI: {NMI:.4f}, HOM: {HOM:.4f}, FMI: {FMI:.4f}')
        return [ARI, AMI, NMI, HOM, FMI]

    else:
        print("Warning: 'cell_type' not found in gex_data.obs. Skipping clustering metrics.")


if __name__ == "__main__":
    # Load single-cell data
    dataset = 'BMMC'
    gex_data = anndata.read_h5ad(
        f'/data2/wfa/project/single_cell_multimodal/data/filter_data/{dataset}/RNA_filter.h5ad')
    print(f"Data shape: {gex_data.X.shape}")

    # split single cell data
    # split_data(dataset_name, gex_data)
    device_list = [6, 6, 7, 7, 0]

    all_folds = 5
    training_function_args = [(gex_data, dataset, fold, device_list[fold]) for fold in range(5)]

    results = multiprocessing_train_fold(5, worker_function, training_function_args, train_fold)
    results = pd.DataFrame(results, columns=['ARI', 'AMI', 'NMI', 'HOM', 'FMI'])
    print(results)

    results.to_csv('RBM_VAE_clustering.csv', index=True)

