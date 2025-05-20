import scvi
import anndata
import scanpy as sc
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from model.utils import load_fold_indices, multiprocessing_train_fold, worker_function, run_pseudotime_on_rep, compute_batch_pseudotime
from model.dataset_param_dict import dataset_params
import os


plt.rcParams.update({
    'figure.titlesize': 7,  # 控制 suptitle 的字体大小
    'axes.titlesize': 7,  # 坐标轴标题字体大小
    'axes.labelsize': 7,  # 坐标轴标签字体大小
    'xtick.labelsize': 6,  # x轴刻度字体大小
    'ytick.labelsize': 6,  # y轴刻度字体大小
    'legend.fontsize': 6,  # 图例字体大小
    'lines.markersize': 6,  # 标记点大小
    'axes.grid': False,  # 默认显示网格
    'axes.linewidth': 0.5,  # 统一设置x轴和y轴宽度（脊线厚度）
    'ytick.major.width': 0.5,  # y轴主刻度线宽度
    'xtick.major.width': 0.5,  # x轴主刻度线宽度
    'ytick.major.size': 2,  # y轴主刻度线长度
    'xtick.major.size': 2,  # x轴主刻度线长度
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def train_xgboost_with_scvi(model, adata, dataset_name, fold_id, params, device_id=6):
    import torch
    import scvi

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
    output_dir = f'result/{dataset_name}/dpt/'
    os.makedirs(output_dir, exist_ok=True)

    # 设置设备
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载交叉验证折索引
    use_cells = ['Erythrocytes', 'Erythroid progenitors', 'HSPCs',
                 'Megakaryocyte progenitors'
                 ]

    # 复制并过滤数据
    unintegrated_adata = adata.copy()
    unintegrated_adata = unintegrated_adata[unintegrated_adata.obs[params['labels_key']].isin(use_cells)].copy()

    # 分批次计算伪时序
    unintegrated_adata = compute_batch_pseudotime(
        unintegrated_adata,
        batch_key=params['batch_key'],
        root_cell_type='HSPCs',
        cluster_key=params['labels_key'],
        rep_key='X_pca'  # 使用PCA嵌入
    )

    # # 构造人工伪时序分数
    # pseudo_order_dict = {
    #     'HSPCs': 0,
    #     'Erythroid progenitors': 1.5,
    #     'Megakaryocyte progenitors': 1,
    #     'Erythrocytes': 2
    # }
    # adata.obs['pseudo_order'] = adata.obs[params['labels_key']].map(pseudo_order_dict).astype(float)
    #
    #
    # # # 为训练和测试数据设置 AnnData
    # if params["batch_key"] != "":
    #     scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=params["batch_key"], labels_key=params["labels_key"])
    #     scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=params["batch_key"], labels_key=params["labels_key"])
    # else:
    #     scvi.model.SCVI.setup_anndata(adata, layer="counts")
    #     scvi.model.SCVI.setup_anndata(adata, layer="counts")
    #
    # # 加载预训练模型
    # model_path = f'models/{dataset_name}/SCVI_model{fold_id}'
    # model = scvi.model.SCVI.load(model_path, adata=adata, device=device_id)
    #
    # adata.obsm['reps'] = model.get_latent_representation(adata)
    # adata = adata[adata.obs[params['labels_key']].isin(use_cells)].copy()
    # adata = run_pseudotime_on_rep(adata,
    #                               rep_key='reps',
    #                               root_cell_type='HSPCs',
    #                               cluster_key=params['labels_key'])
    # # UMAP 用于可视化（不是必须）
    # sc.pp.neighbors(adata, use_rep='reps')
    # sc.tl.umap(adata, min_dist=0.1)
    # sc.pl.umap(
    #     adata,
    #     color=['dpt_pseudotime', params['labels_key'], params['batch_key']],
    #     cmap='viridis',
    #     show=False,
    #     frameon=False,
    #     ncols=1, )
    # plt.savefig(
    #     f'{output_dir}{dataset_name}_SCVI_latent_dpt_fold{fold_id}.pdf',
    #     dpi=1000, bbox_inches='tight')
    # # 匹配细胞名
    # # shared_cells = adata.obs_names.intersection(unintegrated_adata.obs_names)
    #
    # # 取伪时序
    # pt_integrated = adata.obs['dpt_pseudotime']
    # pt_unintegrated = adata.obs['pseudo_order']
    #
    # # 计算 Spearman 相关系数
    # s = pt_integrated.corr(pt_unintegrated, method='spearman')
    #
    # # 转换成 Trajectory Conservation 分数
    # trajectory_conservation_score = (s + 1) / 2
    # print(f"Trajectory conservation score: {trajectory_conservation_score:.3f}")

    # return trajectory_conservation_score


if __name__ == "__main__":
    # Load single-cell data

    dataset_list = dataset_params.keys()
    for dataset_name in dataset_list:

        if dataset_name == 'immune':

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


            multiprocessing_train_fold(5, worker_function, training_function_args, train_xgboost_with_scvi)


