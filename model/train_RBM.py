import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import scanpy as sc
import anndata
from model.DVAE_RBM import scDataset, DVAE_RBM
import torch
from utils import load_fold_indices, split_data
import random


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


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load single-cell data
    dataset_name = 'BMMC'
    gex_data = anndata.read_h5ad(
        f'/data2/wfa/project/single_cell_multimodal/data/filter_data/{dataset_name}/RNA_filter.h5ad')
    print(f"Data shape: {gex_data.X.shape}")

    # split single cell data
    # split_data(dataset_name, gex_data)

    train_idx, test_idx = load_fold_indices(dataset_name, fold_num=0)

    gex_adata_train = gex_data[train_idx, :].copy()
    gex_adata_test = gex_data[test_idx, :].copy()

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
    plt.savefig(f'{dataset_name}_qvae_umap_cell.pdf', dpi=1000, bbox_inches='tight')

    sc.pp.neighbors(gex_adata_test, n_neighbors=10, use_rep='qvae_reps', random_state=42)
    sc.tl.leiden(gex_adata_test, random_state=42)

    # Clustering metrics
    if 'cell_type' in gex_adata_test.obs:
        ARI = metrics.adjusted_rand_score(gex_adata_test.obs['cell_type'], gex_adata_test.obs['leiden'])
        AMI = metrics.adjusted_mutual_info_score(gex_adata_test.obs['cell_type'], gex_adata_test.obs['leiden'])
        NMI = metrics.normalized_mutual_info_score(gex_adata_test.obs['cell_type'], gex_adata_test.obs['leiden'])
        HOM = metrics.homogeneity_score(gex_adata_test.obs['cell_type'], gex_adata_test.obs['leiden'])
        print(f'ARI: {ARI:.4f}, AMI: {AMI:.4f}, NMI: {NMI:.4f}, HOM: {HOM:.4f}')
    else:
        print("Warning: 'cell_type' not found in gex_data.obs. Skipping clustering metrics.")

