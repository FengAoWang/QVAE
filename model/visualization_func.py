import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.rcParams.update({
    # 'figure.figsize': (10, 6),  # 默认图表大小
    'axes.titlesize': 7,  # 坐标轴标题字体大小
    'axes.labelsize': 7,  # 坐标轴标签字体大小
    'xtick.labelsize': 6,  # x轴刻度字体大小
    'ytick.labelsize': 6,  # y轴刻度字体大小
    'legend.fontsize': 6,  # 图例字体大小
    #'lines.linewidth': 0.5,  # 线条宽度
    'lines.markersize': 6,  # 标记点大小
    'axes.grid': False,  # 默认显示网格
    'axes.linewidth': 0.5,  # 统一设置x轴和y轴宽度（脊线厚度）
    'ytick.major.width': 0.5,  # y轴主刻度线宽度
    'xtick.major.width': 0.5,  # x轴主刻度线宽度
    'ytick.major.size': 2,  # y轴主刻度线长度
    'xtick.major.size': 2,  # x轴主刻度线长度
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# def visualize_folds(performance_df_list, vis_cols, methods_list, datasetName):
#     """
#     Visualize the performance of different methods across folds using box plots.
#
#     Parameters:
#     - performance_df_list: List of DataFrames, each containing performance metrics for 5 folds.
#     - vis_cols: List of column names (metrics) to visualize.
#     - methods_list: List of method names corresponding to performance_df_list.
#     """
#     # 输入检查
#     if len(performance_df_list) != len(methods_list):
#         raise ValueError("Length of performance_df_list must match length of methods_list.")
#
#     for df in performance_df_list:
#         for col in vis_cols:
#             if col not in df.columns:
#                 raise ValueError(f"Column '{col}' not found in one of the DataFrames.")
#
#     # 设置绘图风格
#
#     # 计算子图布局
#     n_cols = len(vis_cols)
#     fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 1.5, 3), sharey=False)
#     if n_cols == 1:
#         axes = [axes]  # 确保 axes 是列表，方便统一处理
#
#     # 为每个指标绘制箱线图
#     for idx, metric in enumerate(vis_cols):
#         # 收集所有方法的五折数据
#         data_to_plot = []
#         for method_idx, (df, method) in enumerate(zip(performance_df_list, methods_list)):
#             metric_data = df[metric].dropna().values  # 获取该方法的五折数据
#             data_to_plot.extend([(method, value) for value in metric_data])
#
#         # 转换为 DataFrame 用于 seaborn
#         plot_df = pd.DataFrame(data_to_plot, columns=['Method', metric])
#
#         # 绘制箱线图
#         # Draw box plot with hue to fix palette warning
#         sns.boxplot(
#             x='Method',
#             y=metric,
#             hue='Method',  # Assign x variable to hue
#             data=plot_df,
#             ax=axes[idx],
#             palette='Set2',
#             legend=False  # Disable legend to avoid redundancy
#         )
#
#         # 设置标题和标签
#         axes[idx].set_title(f'{metric} Across Folds')
#         axes[idx].set_xlabel('Method')
#         axes[idx].set_ylabel(metric)
#
#         # 旋转 x 轴标签以避免重叠
#         axes[idx].tick_params(axis='x', rotation=45)
#
#     # 调整布局
#     plt.tight_layout()
#
#     # 显示图形
#     plt.savefig(f'{datasetName}_5_folds_klwarm.png', dpi=1000)
#
#     return fig

# def visualize_folds(performance_df_list, vis_cols, methods_list, datasetName):
#     """
#     Visualize the performance of different methods across folds using box plots in a single figure with vertical lines separating methods.
#
#     Parameters:
#     - performance_df_list: List of DataFrames, each containing performance metrics for 5 folds.
#     - vis_cols: List of column names (metrics) to visualize.
#     - methods_list: List of method names corresponding to performance_df_list.
#     """
#     # 输入检查
#     if len(performance_df_list) != len(methods_list):
#         raise ValueError("Length of performance_df_list must match length of methods_list.")
#
#     for df in performance_df_list:
#         for col in vis_cols:
#             if col not in df.columns:
#                 raise ValueError(f"Column '{col}' not found in one of the DataFrames.")
#
#     # 设置绘图风格
#     # sns.set_style("whitegrid")
#
#     # 创建单个图形
#     fig, ax = plt.subplots(figsize=(len(vis_cols) * len(methods_list) * 0.3, 3))
#
#     # 准备数据
#     data_to_plot = []
#     for method_idx, (df, method) in enumerate(zip(performance_df_list, methods_list)):
#         for metric in vis_cols:
#             metric_data = df[metric].dropna().values
#             for value in metric_data:
#                 data_to_plot.append((metric, method, value))
#
#     # 转换为 DataFrame
#     plot_df = pd.DataFrame(data_to_plot, columns=['Metric', 'Method', 'Value'])
#
#     # 计算每个箱线图的宽度和位置
#     n_methods = len(methods_list)
#     box_width = 0.9 / n_methods  # 每个方法的箱线图宽度
#     group_width = n_methods * box_width
#     metric_positions = np.arange(len(vis_cols)) * (group_width + 0.5)  # 每组的起始位置
#
#     # 绘制箱线图
#     for i, metric in enumerate(vis_cols):
#         for j, method in enumerate(methods_list):
#             # 计算每个箱线图的精确位置
#             pos = metric_positions[i] + j * box_width
#             # 筛选数据
#             metric_data = plot_df[(plot_df['Metric'] == metric) & (plot_df['Method'] == method)]['Value']
#             # 绘制箱线图
#             ax.boxplot(
#                 metric_data,
#                 positions=[pos],
#                 widths=box_width * 0.9,
#                 patch_artist=True,
#                 boxprops=dict(facecolor=sns.color_palette("Set2")[j], color='black'),
#                 medianprops=dict(color='black'),
#                 whiskerprops=dict(color='black'),
#                 capprops=dict(color='black'),
#                 flierprops=dict(marker='o', markersize=5, markerfacecolor=sns.color_palette("Set2")[j])
#             )
#             # # # 在方法之间添加分隔线
#             # if j < n_methods - 1:
#             #     line_pos = pos + box_width
#             #     ax.axvline(x=line_pos, color='gray', linestyle='--', alpha=0.5, ymin=0, ymax=1)
#
#     # 设置 x 轴标签
#     ax.set_xticks(metric_positions + group_width / 2 - box_width / 2)
#     ax.set_xticklabels(vis_cols)
#
#     # 创建图例
#     handles = [plt.Rectangle((0, 0), 1, 1, facecolor=sns.color_palette("Set2")[i]) for i in range(len(methods_list))]
#     ax.legend(handles, methods_list, title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')
#
#     # 设置标题和标签
#     ax.set_title('Performance Across Folds')
#     ax.set_xlabel('Metrics')
#     ax.set_ylabel('Performance')
#
#     # 旋转 x 轴标签
#     # ax.tick_params(axis='x', rotation=45)
#
#     # 保存图形
#     plt.savefig(f'{datasetName}_5_folds_klwarm.png', dpi=1000, bbox_inches='tight')
#
#     return fig

def visualize_folds(performance_df_list, vis_cols, methods_list, datasetName, filename):
    """
    Visualize the performance of different methods across folds using bar plots with error bars in a single figure with vertical lines separating methods.

    Parameters:
    - performance_df_list: List of DataFrames, each containing performance metrics for 5 folds.
    - vis_cols: List of column names (metrics) to visualize.
    - methods_list: List of method names corresponding to performance_df_list.
    """
    # 输入检查
    if len(performance_df_list) != len(methods_list):
        raise ValueError("Length of performance_df_list must match length of methods_list.")

    for df in performance_df_list:
        for col in vis_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in one of the DataFrames.")

    # 创建单个图形
    fig, ax = plt.subplots(figsize=(len(vis_cols) * len(methods_list) * 0.15, 2))

    # 计算每个箱线图的宽度和位置
    n_methods = len(methods_list)
    bar_width = 0.9 / n_methods  # 每个方法的柱状图宽度
    group_width = n_methods * bar_width
    metric_positions = np.arange(len(vis_cols)) * (group_width + 0.2)  # 每组的起始位置

    # 绘制柱状图
    for i, metric in enumerate(vis_cols):
        for j, (method, df) in enumerate(zip(methods_list, performance_df_list)):
            # 计算均值和标准差
            metric_data = df[metric].dropna().values
            mean_value = np.mean(metric_data)
            std_value = np.std(metric_data, ddof=1)  # 使用样本标准差
            # 计算柱状图的精确位置
            pos = metric_positions[i] + j * bar_width
            # 绘制柱状图
            ax.bar(
                pos,
                mean_value,
                yerr=std_value,  # 添加误差棒（标准差）
                width=bar_width * 0.9,
                color=sns.color_palette("Paired")[j],
                edgecolor='black',
                linewidth=0.3,  # 设置描边宽度（可调整）
                error_kw=dict(lw=0.3, capsize=1, capthick=0.3),  # 误差棒设置
                label=method if i == 0 else None  # 仅为第一个指标添加图例
            )
            # # 在方法之间添加分隔线
            # if j < n_methods - 1:
            #     line_pos = pos + bar_width
            #     ax.axvline(x=line_pos, color='gray', linestyle='--', alpha=0.5, ymin=0, ymax=1)

    # 设置 x 轴标签
    ax.set_xticks(metric_positions + group_width / 2 - bar_width / 2)
    ax.set_xticklabels(vis_cols)

    # 创建图例
    ax.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 设置标题和标签
    ax.set_title('Performance Across Folds')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Performance')

    # 设置 y 轴范围，从 0.3 开始
    ax.set_ylim(bottom=0.2)

    os.makedirs('figures/clustering_performance/', exist_ok=True)
    # 保存图形
    plt.savefig(f'figures/clustering_performance/{datasetName}_{filename}.pdf', dpi=1000, bbox_inches='tight')

    return fig



def visualize_latentDims(df_list, vis_cols, latent_dims, datasetName, methodsList, x_dim_name, figure_name):
    if len(methodsList) == 1:
        figure_name = figure_name + methodsList[0]
    if len(methodsList) > 1:
        figure_name = figure_name + 'comparison'

    if len(vis_cols) == 1:
        figure_name = figure_name + vis_cols[0]

    # Set up the plot
    plt.figure(figsize=(2, 1.5))  # Slightly larger for clarity with multiple methods

    # Generate unique colors for each method-metric combination
    total_combinations = len(methodsList) * len(vis_cols)
    colors = sns.color_palette("husl", total_combinations)

    # Initialize color index
    color_idx = 0

    # Process each method's dataframe list
    for method_idx, (method, dfs) in enumerate(zip(methodsList, df_list)):
        # Add latent_dim column to each dataframe
        for i, df in enumerate(dfs):
            df[x_dim_name] = latent_dims[i]

        # Combine dataframes for this method
        combined_df = pd.concat(dfs)

        # Calculate mean and std for each metric
        summary_df = combined_df.groupby(x_dim_name)[vis_cols].agg(['mean', 'std']).reset_index()

        # Plot lines for each metric for this method
        for col in vis_cols:
            # Plot mean line with unique color for method-metric pair
            if len(methodsList) == 1:
                plt.plot(summary_df[x_dim_name], summary_df[col]['mean'],
                         marker='o', color=colors[color_idx],
                         label=f'{col}')

            if len(methodsList) > 1:
                plt.plot(summary_df[x_dim_name], summary_df[col]['mean'],
                         marker='o', color=colors[color_idx],
                         label=f'{method} {col}')
            # Plot error band with matching color
            plt.fill_between(summary_df[x_dim_name],
                             summary_df[col]['mean'] - summary_df[col]['std'],
                             summary_df[col]['mean'] + summary_df[col]['std'],
                             color=colors[color_idx], alpha=0.1)
            color_idx += 1  # Move to next color for next method-metric pair

    # Customize plot
    plt.xlabel(x_dim_name)
    plt.ylabel('Value')
    plt.title(f'{dataset_name}')
    plt.xticks(latent_dims, latent_dims, rotation=40)  # Uniform x-ticks
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(f'figures/clustering_latent/{datasetName}_{x_dim_name}_{figure_name}.pdf', dpi=1000, bbox_inches='tight')



if __name__ == '__main__':

    #   benchmark
    # dataset_name = 'BMMC_multiome'
    # RBM_VAE_df = pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim256_layernorm_batchSize2048_weight_decay.csv')
    # VAE_df = pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim128_weight_decay.csv')
    # SCVI_df = pd.read_csv(f'result/{dataset_name}/SCVI_{dataset_name}_clustering.csv')
    # tr_VAE_df = pd.read_csv(f'result/{dataset_name}/trVAE_{dataset_name}_clustering.csv')
    # LDVAE_df = pd.read_csv(f'result/{dataset_name}/LDVAE_{dataset_name}_clustering.csv')
    # AUTOZI_df = pd.read_csv(f'result/{dataset_name}/AUTOZI_{dataset_name}_clustering.csv')
    #
    #
    # # visualize_folds([RBM_VAE_df,  RBM_VAE_df2, RBM_VAE_df3, SCVI_df], ['ARI', 'AMI', 'NMI', 'HOM', 'FMI'], ['RBM_VAE', 'RBM_VAE_wd_layer_bs2048', 'RBM_VAE_wd', 'SCVI'], dataset_name, 'bio_conservation')
    #
    # visualize_folds([RBM_VAE_df,  SCVI_df, tr_VAE_df, LDVAE_df, AUTOZI_df, VAE_df],
    #                 ['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI'],
    #                 ['RBM_VAE', 'SCVI', 'trVAE', 'LDVAE', 'AUTOZI', 'VAE'],
    #                 dataset_name,
    #                 'bio_conservation_bm')
    # visualize_folds([RBM_VAE_df,  SCVI_df, tr_VAE_df, LDVAE_df, AUTOZI_df, VAE_df],
    #                 ['iLISI', 'KBET', 'Graph connectivity', 'Batch correction'],
    #                 ['RBM_VAE', 'SCVI', 'trVAE', 'LDVAE', 'AUTOZI', 'VAE'],
    #                 dataset_name,
    #                 'batch_remove_bm')

    #
    # RBM_df_list = []
    # VAE_df_list = []
    # latent_dims = [16, 32, 64, 128, 256, 512]
    # methods = ['RBM-VAE']
    #
    # dataset_name = 'HLCA_core'
    # for latent_dim in latent_dims:
    #     RBM_df_list.append(pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim{latent_dim}.csv'))
    # # for latent_dim in latent_dims:
    # #     VAE_df_list.append(pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim{latent_dim}.csv'))
    # df_list = [RBM_df_list]
    #
    # visualize_latentDims(df_list, ['ARI', 'AMI', 'NMI', 'HOM', 'FMI'], latent_dims, dataset_name, methods)

    # RBM_df_list = []
    # VAE_df_list = []
    # latent_dims = [16, 32, 64, 128, 256, 512]
    # methods = ['RBM-VAE']
    #
    # dataset_name = 'pancreas'
    # for latent_dim in latent_dims:
    #     RBM_df_list.append(pd.read_csv(f'result/{dataset_name}/RBM_VAE_pancreas_clustering_latentDim256_layernorm_batchSize{}_weight_decay.csv'))
    #
    # # for latent_dim in latent_dims:
    # #     VAE_df_list.append(pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim{latent_dim}_weight_decay.csv'))
    #
    # # for latent_dim in latent_dims:
    # #     RBM_df_list2.append(pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim{latent_dim}_weight_decay.csv'))
    # df_list = [RBM_df_list]
    #
    # for col in ['ARI', 'AMI', 'NMI', 'HOM', 'FMI']:
    #     visualize_latentDims(df_list, [col], latent_dims, dataset_name, methods)


    RBM_df_list = []
    VAE_df_list = []
    batch_dims = [128, 256, 512, 1024, 2048]
    methods = ['RBM-VAE']

    dataset_name = 'immune'
    for batch_dim in batch_dims:
        RBM_df_list.append(pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim256_layernorm_batchSize{batch_dim}_weight_decay.csv'))

    # for latent_dim in latent_dims:
    #     VAE_df_list.append(pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim{latent_dim}_weight_decay.csv'))

    # for latent_dim in latent_dims:
    #     RBM_df_list2.append(pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim{latent_dim}_weight_decay.csv'))
    df_list = [RBM_df_list]

    # for col in ['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI']:
    visualize_latentDims(df_list, ['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI'], batch_dims, dataset_name, methods, 'batch size', 'layernorm')
