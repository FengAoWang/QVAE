import scvi
import anndata
import scanpy as sc
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from model.utils import load_fold_indices, multiprocessing_train_fold, worker_function
from model.dataset_param_dict import dataset_params
import os



def train_xgboost_with_scvi(model, adata, dataset_name, fold_id, params, device_id=6):
    import torch
    import xgboost as xgb
    import seaborn as sns
    import scvi
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import LabelEncoder
    from model.visualization_func import plot_confusion_matrix

    """
    加载预训练的 SCVI 模型，冻结参数，提取嵌入，训练 XGBoost 分类器并在测试集上验证

    参数:
        adata: AnnData 对象，包含基因表达数据
        dataset_name: 数据集名称
        fold_id: 交叉验证折编号
        params: 参数字典，包含 batch_key, labels_key 等
        device_id: GPU 设备 ID，默认为 6

    返回:
        classification_metrics: 分类性能指标列表 [accuracy, precision, recall, f1]
    """
    # 设置输出目录
    output_dir = f'result/{dataset_name}/classification_metrics'
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载交叉验证折索引
    train_idx, test_idx = load_fold_indices(dataset_name, fold_num=fold_id)

    # 划分训练和测试数据
    adata_train = adata[train_idx, :].copy()
    adata_test = adata[test_idx, :].copy()

    print(f"训练数据形状: {adata_train.X.shape}")
    print(f"测试数据形状: {adata_test.X.shape}")

    # # 为训练和测试数据设置 AnnData
    if params["batch_key"] != "":
        scvi.model.SCVI.setup_anndata(adata_train, layer="counts", batch_key=params["batch_key"], labels_key=params["labels_key"])
        scvi.model.SCVI.setup_anndata(adata_test, layer="counts", batch_key=params["batch_key"], labels_key=params["labels_key"])
    else:
        scvi.model.SCVI.setup_anndata(adata_train, layer="counts")
        scvi.model.SCVI.setup_anndata(adata_test, layer="counts")

    # 加载预训练模型
    model_path = f'models/{dataset_name}/SCVI_model{fold_id}'
    model = scvi.model.SCVI.load(model_path, adata=adata_train, device=device_id)

    # 提取训练集和测试集的嵌入
    latent_train = model.get_latent_representation(adata_train)
    latent_test = model.get_latent_representation(adata_test)

    adata_train.obsm['SCVI_latent'] = latent_train
    adata_test.obsm['SCVI_latent'] = latent_test

    # 获取嵌入和标签
    X_train = adata_train.obsm['SCVI_latent']
    X_test = adata_test.obsm['SCVI_latent']

    # 确保标签存在
    if params['labels_key'] not in adata_train.obs:
        raise ValueError(f"标签键 {params['labels_key']} 未在 adata.obs 中找到")

    # 编码标签
    le = LabelEncoder()
    y_train = le.fit_transform(adata_train.obs[params['labels_key']])
    y_test = le.transform(adata_test.obs[params['labels_key']])

    # 初始化XGBoost分类器
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist',  # 使用基于直方图的算法，支持GPU
        device='cuda',  # 明确指定使用GPU
        predictor='cuda',  # 使用GPU预测器
        # gpu_id=device_id,  # 指定GPU设备ID，与神经网络一致
        n_jobs=-1
    )

    # 训练 XGBoost 分类器
    xgb_classifier.fit(X_train, y_train)

    # 在测试集上预测
    y_pred = xgb_classifier.predict(X_test)

    # 计算分类指标
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    classification_metrics = [accuracy, precision, recall, f1]

    labels = le.classes_

    plot_confusion_matrix(y_test, y_pred, labels, dataset_name, fold_id, output_dir, 'scvi')

    # 保存分类结果
    # results_file = f'{output_dir}{dataset_name}_SCVI_XGBoost_classification_fold{fold_id}.txt'
    # with open(results_file, 'w') as f:
    #     f.write(f"准确率: {accuracy:.4f}\n")
    #     f.write(f"精确率: {precision:.4f}\n")
    #     f.write(f"召回率: {recall:.4f}\n")
    #     f.write(f"F1 分数: {f1:.4f}\n")

    return classification_metrics


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

        device_list = [7, 1, 4, 5, 7]

        all_folds = 5
        training_function_args = [('SCVI', gex_data, dataset_name, fold, dataset_params[dataset_name], device_list[fold]) for fold in range(5)]


        results = multiprocessing_train_fold(5, worker_function, training_function_args, train_xgboost_with_scvi)
        results = pd.DataFrame(results, columns=['acc', 'pre', 'recall', 'f1'])

        print(results)

        results.to_csv(f'result/{dataset_name}/classification_metrics/SCVI_{dataset_name}_cell_classify.csv', index=True)
