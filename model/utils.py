from sklearn.model_selection import KFold
import os
import pickle
from sklearn import metrics
import torch.multiprocessing as mp
import logging
import matplotlib.pyplot as plt
import torch
import scanpy as sc
from scipy.stats import entropy
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection


def split_data(dataset, adata):
    # 设置5折交叉验证
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 创建保存目录
    output_dir = f'split_indices/{dataset}/'
    os.makedirs(output_dir, exist_ok=True)

    # 获取样本数量
    n_samples = adata.X.shape[0]

    # 初始化字典来存储所有折的索引
    all_folds = {
        'train': [],
        'test': []
    }

    # 进行5折划分
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(n_samples))):
        # 存储索引
        all_folds['train'].append(train_idx)
        all_folds['test'].append(test_idx)

        # 打印每折的索引数量
        print(f"Fold {fold + 1}:")
        print(f"Train indices: {len(train_idx)}")
        print(f"Test indices: {len(test_idx)}")

    # 保存所有折的索引到一个pickle文件
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

    # 验证加载的数据
    print(f"Loaded indices for dataset: {dataset}")
    print(f"Number of folds: {len(all_folds['train'])}")

    if fold_num is not None:
        # 验证fold_num有效性
        if not 0 <= fold_num < len(all_folds['train']):
            raise ValueError(f"fold_num must be between 0 and {len(all_folds['train']) - 1}")

        # 返回特定fold的训练和测试索引
        train_indices = all_folds['train'][fold_num]
        test_indices = all_folds['test'][fold_num]
        print(f"Fold {fold_num}:")
        print(f"Train indices count: {len(train_indices)}")
        print(f"Test indices count: {len(test_indices)}")
        return train_indices, test_indices

    # 如果未指定fold_num，返回所有folds
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


def compute_batchEffect(adata,
                        batch_key='batch',
                        label_key='cell_type',
                        x_emb='reps'):
    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=[x_emb],
        batch_correction_metrics=BatchCorrection(),
        n_jobs=6,
    )

    bm.benchmark()
    df = bm.get_results(min_max_scale=False)
    print(df.columns)

    # 筛选批次矫正相关的指标

    # ['Isolated labels', 'KMeans NMI', 'KMeans ARI', 'Silhouette label',
    #        'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity',
    #        'PCR comparison', 'Batch correction', 'Bio conservation', 'Total']
    # 只取出 'reps' 对应的行
    reps_metrics = df.loc[x_emb]
    print(reps_metrics.values)
    return reps_metrics.values


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



def train_fold(Model, adata, dataset_name, fold_id, DataParams, TrainingParams, device_id=6):
    output_dir = f'result/{dataset_name}/'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_idx, test_idx = load_fold_indices(dataset_name, fold_num=fold_id)

    gex_adata_train = adata[train_idx, :].copy()
    gex_adata_test = adata[test_idx, :].copy()

    print(f"Data shape: {gex_adata_train.X.shape}")

    model = Model(latent_dim=TrainingParams['latent_dim'],
                  device=device,
                  use_norm=TrainingParams['normaliztion'])
    model.set_adata(gex_adata_train, batch_key=DataParams['batch_key'])

    model.fit(gex_adata_train,
              epochs=150,
              lr=TrainingParams['lr'],
              early_stopping_patience=10,
              n_epochs_kl_warmup=50,
              batch_size=TrainingParams['batch_size'],
              )

    # Add latent representations to AnnData
    gex_adata_test.obsm['reps'] = model.get_representation(gex_adata_test)
    latent_test = gex_adata_test.obsm['reps']

    # UMAP and clustering
    sc.pp.neighbors(gex_adata_test, n_neighbors=10, use_rep='reps')
    sc.tl.umap(gex_adata_test)
    # plt.figure(figsize=(8, 3))
    sc.pl.umap(gex_adata_test, color=[DataParams['labels_key'], DataParams['batch_key']], show=False)
    plt.savefig(f'{output_dir}{dataset_name}_{Model.__name__}_cell_latentDim{TrainingParams["latent_dim"]}_fold{fold_id}.png', dpi=1000, bbox_inches='tight')

    sc.pp.neighbors(gex_adata_test, n_neighbors=10, use_rep='reps', random_state=42)
    sc.tl.leiden(gex_adata_test, random_state=42)
    sc.tl.louvain(gex_adata_test, random_state=42)

    # Clustering metrics
    if DataParams['labels_key'] in gex_adata_test.obs:
        leiden_ARI, leiden_AMI, leiden_NMI, leiden_HOM, leiden_FMI = compute_clusters_performance(gex_adata_test, DataParams['labels_key'])
        louvain_ARI, louvain_AMI, louvain_NMI, louvain_HOM, louvain_FMI = compute_clusters_performance(gex_adata_test, DataParams['labels_key'], cluster_key='louvain')

        # scib_values = compute_batchEffect(gex_adata_test, DataParams['batch_key'], DataParams['labels_key'],
        #                                   x_emb='reps')
        clustering_value = [leiden_ARI, leiden_AMI, leiden_NMI, leiden_HOM, leiden_FMI, louvain_ARI, louvain_AMI, louvain_NMI, louvain_HOM, louvain_FMI]

        # clustering_value.extend(scib_values)
        return clustering_value

    else:
        print("Warning: 'cell_type' not found in gex_data.obs. Skipping clustering metrics.")
