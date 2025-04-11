from sklearn.model_selection import KFold
import os
import pickle


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

