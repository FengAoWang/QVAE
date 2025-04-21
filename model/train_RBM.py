import numpy as np
import pandas as pd
import anndata
from DVAE_RBM import scDataset, DVAE_RBM
import torch
from utils import load_fold_indices, split_data, multiprocessing_train_fold, worker_function, train_fold
import random
from dataset_param_dict import dataset_params, training_params


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
    # Load single-cell data
    dataset_list = dataset_params.keys()
    training_params_list = training_params.keys()

    for dataset_name in dataset_list:
        for training_param in training_params_list:
            print(dataset_name)
            gex_data = anndata.read_h5ad(dataset_params[dataset_name]['file_path'])
            print(f"Data shape: {gex_data.X.shape}")
            print(gex_data.obs.columns)

            # split single cell data
            # split_data(dataset_name, gex_data)
            #
            device_list = [6, 6, 7, 7, 0]

            training_function_args = [(DVAE_RBM,
                                       gex_data,
                                       dataset_name,
                                       fold,
                                       dataset_params[dataset_name],
                                       training_params[training_param],
                                       device_list[fold]) for fold in range(5)]
            # train_fold(DVAE_RBM, gex_data, dataset_name, 0, dataset_params[dataset_name], training_params[training_param])

            results = multiprocessing_train_fold(5, worker_function, training_function_args, train_fold)
            results = pd.DataFrame(results, columns=['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_HOM', 'leiden_FMI',
                                                     'louvain_ARI', 'louvain_AMI', 'louvain_NMI', 'louvain_HOM', 'louvain_FMI',])
                                                 # 'Isolated labels', 'KMeans NMI', 'KMeans ARI', 'Silhouette label',
                                                 # 'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity',
                                                 # 'PCR comparison', 'Batch correction', 'Bio conservation', 'Total'])
            print(results)

            results.to_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim256_{training_params[training_param]["normaliztion"]}_batchSize{training_params[training_param]["batch_size"]}_weight_decay.csv', index=True)
