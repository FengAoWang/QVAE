import numpy as np
import pandas as pd
import anndata
from DVAE_RBM import DVAE_RBM
import torch
from utils import split_data, multiprocessing_train_fold, worker_function, train_fold, GridSearchConfig
import random
from dataset_param_dict import dataset_params
import os


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


def set_grid_configs():
    grid_configs = []
    for sample_method in ["ising_fsa"]:
        for norm_method in ["batch"]:
            for latent_dim in [16, 32, 64, 128, 256]:
                config = GridSearchConfig(normalization_method=norm_method,
                                          sample_method=sample_method,
                                          latent_dim=latent_dim)
                grid_configs.append(config)

    return grid_configs


if __name__ == "__main__":
    # Load single-cell data
    output_name = "result"
    dataset_list = dataset_params.keys()
    for dataset_name in dataset_list:
        # if dataset_name == 'HLCA_core':

        print(dataset_name)
        gex_data = anndata.read_h5ad(dataset_params[dataset_name]['file_path'])
        print(f"Data shape: {gex_data.X.shape}")
        print(gex_data.obs.columns)

        # split single cell data
        split_data(dataset_name, gex_data)

        grid_configs = set_grid_configs()
        for config in grid_configs:
            # 设置随机种子（确保实验可重复）
            set_seed(config.seed)

            # 构造保存结果的文件夹名称
            config_folder = str(config)
            output_dir = os.path.join(output_name, dataset_name, config_folder)
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nTraining with config: {config_folder}")

            # 分配设备（示例：按fold循环分配）
            device_list = [0, 0, 0, 0, 0]  # 可根据实际GPU情况调整

            # 构造训练参数列表（将config传递给每个fold）
            training_function_args = [
                (DVAE_RBM, gex_data, dataset_name, fold, dataset_params[dataset_name], config, output_name,
                 device_list[fold], 1)
                for fold in range(5)
            ]

            results = multiprocessing_train_fold(5, worker_function, training_function_args, train_fold)

            # 保存结果
            results_df = pd.DataFrame(results, columns=['ARI', 'AMI', 'NMI', 'HOM', 'FMI'])
            results_df.to_csv(os.path.join(output_dir, f'{dataset_name}_clustering.csv'), index=False)
            print(f"Results saved to {output_dir}")
