import scvi
import anndata
import scanpy as sc
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from model.utils import load_fold_indices, split_data, compute_clusters_performance, multiprocessing_train_fold, worker_function, compute_batchEffect
import torch
import pytorch_lightning as pl
from model.dataset_param_dict import dataset_params
import os
import numpy as np
import random


# scvi.settings.seed = 0
print("Last run with scvi-tools version:", scvi.__version__)
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

# set_seed(3407)

def train_fold(Model, adata, dataset_name, fold_id, params, device_id=6):

    output_dir = f'result/{dataset_name}/integration/SCVI/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'models/{dataset_name}/', exist_ok=True)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if params["batch_key"] != "":
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=params["batch_key"], labels_key=params["labels_key"])
    else:
        scvi.model.SCVI.setup_anndata(adata, layer="counts")

    # Define the range of epochs to iterate through
    model = scvi.model.SCVI(adata, n_layers=2, gene_likelihood="nb")
    # Configure the trainer to use the specified device
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[device_id] if torch.cuda.is_available() else None,
    )
    model.train(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[device_id] if torch.cuda.is_available() else None,
    )

    model.save(f'models/{dataset_name}/SCVI_model{fold_id}')

    # Get latent representation
    latent_test = model.get_latent_representation(adata)
    adata.obsm['SCVI_latent'] = latent_test

    PEAKVI_LATENT_KEY = "SCVI_latent"

    # Compute neighbors and UMAP
    sc.pp.neighbors(adata, use_rep=PEAKVI_LATENT_KEY)
    sc.tl.umap(adata, min_dist=0.2)
    sc.tl.leiden(adata, random_state=42)

    # Save UMAP plot
    # plt.figure(figsize=(12, 3))
    colors_use = [params['labels_key'], params['batch_key'], 'leiden'] if params['batch_key'] != "" else [params['labels_key'], 'leiden']
    sc.pl.umap(adata,
               color=colors_use,
               show=False,
               frameon=False,
               ncols=1,
               )
    plt.savefig(f'{output_dir}{dataset_name}_SCVI_cell_{fold_id}.pdf', bbox_inches='tight', dpi=1000)

    # sc.pl.spatial(adata_test,
    #               color=colors_use,
    #               spot_size=1.0,
    #               show=False,
    #               frameon=False,
    #               ncols=1)
    # plt.savefig(
    #     f'{output_dir}{dataset_name}_SCVI_Spatial_cell_{fold_id}.pdf',
    #     dpi=1000, bbox_inches='tight')

    # plt.close()  # Close the plot to free memory

    # sc.tl.louvain(adata_test, random_state=42)

    # Clustering metrics
    clustering_value = []
    if params['labels_key'] in adata.obs:
        leiden_ARI, leiden_AMI, leiden_NMI, leiden_HOM, leiden_FMI = compute_clusters_performance(adata,
                                                                                                  params['labels_key'])
        clustering_value.extend([leiden_ARI, leiden_AMI, leiden_NMI, leiden_HOM, leiden_FMI])

        # louvain_ARI, louvain_AMI, louvain_NMI, louvain_HOM, louvain_FMI = compute_clusters_performance(gex_adata_test, DataParams['labels_key'], cluster_key='louvain')

    # if params['batch_key'] in adata_test.obs:
    #
    #     scib_values = compute_batchEffect(adata_test, params['batch_key'], params['labels_key'],
    #                                       x_emb='SCVI_latent')
    #     clustering_value.extend(scib_values)

    return clustering_value


if __name__ == "__main__":
    # Load single-cell data

    dataset_list = dataset_params.keys()
    for dataset_name in dataset_list:

        # if dataset_name == 'pancreas':

        gex_data = anndata.read_h5ad(dataset_params[dataset_name]['file_path'])

        # gex_data.X = gex_data.layers['counts']
        # print(f"Data shape: {gex_data.X.shape}")
        # print(gex_data.layers['counts'])

        # gex_data.X = gex_data.layers['counts']

        # split single cell data
        # split_data(dataset_name, gex_data)

        device_list = [0, 1, 4, 5, 2]

        all_folds = 5
        training_function_args = [('SCVI', gex_data, dataset_name, fold, dataset_params[dataset_name], device_list[fold]) for fold in range(5)]

        results = multiprocessing_train_fold(5, worker_function, training_function_args, train_fold)
        results = pd.DataFrame(results, columns=['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_HOM', 'leiden_FMI',
                                                     # 'louvain_ARI', 'louvain_AMI', 'louvain_NMI', 'louvain_HOM', 'louvain_FMI',
                                                 # 'Isolated labels', 'KMeans NMI', 'KMeans ARI', 'Silhouette label',
                                                 # 'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity',
                                                 # 'PCR comparison', 'Batch correction', 'Bio conservation', 'Total'
                                                 ])

        print(results)
        results.to_csv(f'result/{dataset_name}/integration/SCVI_{dataset_name}_cell_clustering.csv', index=True)
