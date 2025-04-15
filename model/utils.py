from sklearn.model_selection import KFold
import os
import pickle
from sklearn import metrics
import torch.multiprocessing as mp
import logging
import matplotlib.pyplot as plt
import torch
import scanpy as sc


class GridSearchConfig:
    def __init__(self,
                 hidden_dim=512,
                 latent_dim=256,
                 beta=0.5,
                 beta_kl=0.0001,
                 normalization_method="batch",  # "batch" or "layer"
                 sample_method="gibbs",  # "gibbs" or "ising_noise" or "ising_fsa"
                 lr=1e-4,
                 rbm_lr=1e-3,
                 epochs=200,
                 batch_size=128,
                 early_stopping_patience=10,
                 seed=42):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.beta_kl = beta_kl
        self.normalization_method = normalization_method
        self.sample_method = sample_method
        self.lr = lr
        self.rbm_lr = rbm_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

    def __str__(self):
        return (f"norm-{self.normalization_method}_"
                f"sample-{self.sample_method}_"
                f"latent-{self.latent_dim}")


def split_data(dataset, adata):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    output_dir = f'split_indices/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)

    n_samples = adata.X.shape[0]

    all_folds = {
        'train': [],
        'test': []
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(range(n_samples))):
        all_folds['train'].append(train_idx)
        all_folds['test'].append(test_idx)

        print(f"Fold {fold + 1}:")
        print(f"Train indices: {len(train_idx)}")
        print(f"Test indices: {len(test_idx)}")

    output_file = f'{output_dir}/five_fold_indices.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_folds, f)


def load_fold_indices(dataset, fold_num=None):
    """
    Load fold indices from pickle file.

    Parameters:
    - dataset (str): Name of the dataset
    - fold_num (int, optional): Specific fold number to load (0-4). If None, returns all folds

    Returns:
    - If fold_num is specified: tuple of (train_indices, test_indices)
    - If fold_num is None: dictionary containing all folds
    """
    # 设置文件路径
    input_file = f'split_indices/{dataset}/five_fold_indices.pkl'

    # 读取pickle文件
    with open(input_file, 'rb') as f:
        all_folds = pickle.load(f)

    print(f"Loaded indices for dataset: {dataset}")
    print(f"Number of folds: {len(all_folds['train'])}")

    if fold_num is not None:
        if not 0 <= fold_num < len(all_folds['train']):
            raise ValueError(f"fold_num must be between 0 and {len(all_folds['train']) - 1}")

        train_indices = all_folds['train'][fold_num]
        test_indices = all_folds['test'][fold_num]
        print(f"Fold {fold_num}:")
        print(f"Train indices count: {len(train_indices)}")
        print(f"Test indices count: {len(test_indices)}")
        return train_indices, test_indices

    return all_folds


def compute_clusters_performance(adata,
                                 cell_key,
                                 cluster_key='leiden'):
    ARI = metrics.adjusted_rand_score(adata.obs[cell_key], adata.obs[cluster_key])
    AMI = metrics.adjusted_mutual_info_score(adata.obs[cell_key], adata.obs[cluster_key])
    NMI = metrics.normalized_mutual_info_score(adata.obs[cell_key], adata.obs[cluster_key])
    HOM = metrics.homogeneity_score(adata.obs[cell_key], adata.obs[cluster_key])
    FMI = metrics.fowlkes_mallows_score(adata.obs[cell_key], adata.obs[cluster_key])
    return ARI, AMI, NMI, HOM, FMI


def multiprocessing_train_fold(folds, worker_function, func_args_list, train_function):
    processes = []
    return_queue = mp.Queue()
    for i in range(folds):
        p = mp.Process(target=worker_function, args=(func_args_list[i], return_queue, train_function))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results_dict = {}
    for _ in range(folds):
        fold_id, result = return_queue.get()
        if result is None:
            logging.error(f"Fold {fold_id} failed.")
        results_dict[fold_id] = result
    print(results_dict)

    results = [results_dict[i] for i in range(folds)]
    return results


def worker_function(func_args, return_queue, train_function):
    fold_id = func_args[3]
    try:
        result = train_function(*func_args)
        return_queue.put((fold_id, result))
    except Exception as e:
        logging.error(f"Error in fold {fold_id}: {str(e)}")
        return_queue.put((fold_id, None))


def train_fold(Model, adata, dataset_name, fold_id, params, config, output_name, device_id=0, verbose=0):
    config_folder = str(config)
    output_dir = os.path.join(output_name, dataset_name, config_folder)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_idx, test_idx = load_fold_indices(dataset_name, fold_num=fold_id)
    gex_adata_train = adata[train_idx, :].copy()
    gex_adata_test = adata[test_idx, :].copy()
    intermediate_results = None

    if Model.__name__ == "DVAE_RBM":
        model = Model(
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            beta=config.beta,
            beta_kl=config.beta_kl,
            normalization_method=config.normalization_method,
            sample_method=config.sample_method,
            device=device
        )
        model.set_adata(gex_adata_train, batch_key=params['batch_key'])

        intermediate_results = model.fit(
            gex_adata_train,
            epochs=config.epochs,
            lr=config.lr,
            rbm_lr=config.rbm_lr,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            verbose=verbose
        )



    elif Model.__name__ == "VAE":
        model = Model(
            latent_dim=config.latent_dim,
            device=device
        )
        model.set_adata(gex_adata_train, batch_key=params['batch_key'])

        intermediate_results = model.fit(
            gex_adata_train,
            epochs=config.epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            early_stopping_patience=config.early_stopping_patience,
            verbose=verbose
        )

    if verbose == 1 and intermediate_results is not None:
        with open(f'{output_dir}/{dataset_name}_{Model.__name__}_fold{fold_id}_intermediate_results.pkl', 'wb') as f:
            pickle.dump(intermediate_results, f)


    gex_adata_test.obsm['qvae_reps'] = model.get_representation(gex_adata_test)
    latent_test = gex_adata_test.obsm['qvae_reps']

    sc.pp.neighbors(gex_adata_test, n_neighbors=10, use_rep='qvae_reps')
    sc.tl.umap(gex_adata_test)
    sc.pl.umap(
        gex_adata_test,
        color=[params['labels_key'], params['batch_key']],
        show=False,
        title=f"Config: {config_folder} (Fold {fold_id})"
    )

    plt.savefig(f'{output_dir}/{dataset_name}_{Model.__name__}_cell_{fold_id}.png', dpi=1000, bbox_inches='tight')

    sc.pp.neighbors(gex_adata_test, n_neighbors=10, use_rep='qvae_reps', random_state=42)
    sc.tl.leiden(gex_adata_test, random_state=42)

    if params['labels_key'] in gex_adata_test.obs:
        ARI, AMI, NMI, HOM, FMI = compute_clusters_performance(gex_adata_test, params['labels_key'])
        print(f'Fold {fold_id} Metrics: ARI={ARI:.4f}, AMI={AMI:.4f}, NMI={NMI:.4f}')
        return [ARI, AMI, NMI, HOM, FMI]
    else:
        print("Warning: Missing labels. Skipping metrics.")
        return [None] * 5
