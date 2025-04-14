import scvi
import anndata
import scanpy as sc
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from utils import load_fold_indices, split_data, compute_clusters_performance, multiprocessing_train_fold, worker_function
import torch
import pytorch_lightning as pl
from dataset_param_dict import dataset_params
import os

scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)


def train_fold(Model, adata, dataset_name, fold_id, params, device_id=6):

    output_dir = f'result/{dataset_name}/'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_idx, test_idx = load_fold_indices(dataset_name, fold_num=fold_id)

    adata_train = adata[train_idx, :].copy()
    adata_test = adata[test_idx, :].copy()

    scvi.model.SCVI.setup_anndata(adata_train, layer="counts", batch_key=params["batch_key"], labels_key=params["labels_key"])

    # Define the range of epochs to iterate through
    model = scvi.model.SCVI(adata_train)
    # Configure the trainer to use the specified device
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[device_id] if torch.cuda.is_available() else None,
    )
    model.train(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[device_id] if torch.cuda.is_available() else None,
    )

    # Get latent representation
    latent_test = model.get_latent_representation(adata=adata_test)
    adata_test.obsm['SCVI_latent'] = latent_test

    PEAKVI_LATENT_KEY = "SCVI_latent"

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata_test, use_rep=PEAKVI_LATENT_KEY)
    sc.tl.umap(adata_test, min_dist=0.2)

    # Save UMAP plot
    # plt.figure(figsize=(12, 3))
    sc.pl.umap(adata_test, color=[params['labels_key'], params['batch_key']], show=False)
    plt.savefig(f'{output_dir}{dataset_name}_SCVI_cell_{fold_id}.png', bbox_inches='tight', dpi=1000)
    plt.close()  # Close the plot to free memory

    sc.pp.neighbors(adata_test, n_neighbors=10, use_rep=PEAKVI_LATENT_KEY, random_state=42)
    sc.tl.leiden(adata_test, random_state=42)

    ARI, AMI, NMI, HOM, FMI = compute_clusters_performance(adata_test, params["labels_key"])
    print(f'ARI: {ARI:.4f}, AMI: {AMI:.4f}, NMI: {NMI:.4f}, HOM: {HOM:.4f}, FMI: {FMI:.4f}')

    return [ARI, AMI, NMI, HOM, FMI]


if __name__ == "__main__":
    # Load single-cell data

    dataset_list = dataset_params.keys()
    for dataset_name in dataset_list:

        # if dataset_name == 'pancreas':

        gex_data = anndata.read_h5ad(dataset_params[dataset_name]['file_path'])

        gex_data.X = gex_data.layers['counts']
        print(f"Data shape: {gex_data.X.shape}")
        print(gex_data.layers['counts'])

        # gex_data.X = gex_data.layers['counts']

        # split single cell data
        # split_data(dataset_name, gex_data)
        device_list = [0, 1, 2, 3, 5]

        all_folds = 5
        training_function_args = [('SCVI', gex_data, dataset_name, fold, dataset_params[dataset_name], device_list[fold]) for fold in range(5)]

        results = multiprocessing_train_fold(5, worker_function, training_function_args, train_fold)
        results = pd.DataFrame(results, columns=['ARI', 'AMI', 'NMI', 'HOM', 'FMI'])
        print(results)

        results.to_csv(f'result/{dataset_name}/SCVI_{dataset_name}_clustering.csv', index=True)

