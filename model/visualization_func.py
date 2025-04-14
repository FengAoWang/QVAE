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

def visualize_folds(performance_df_list, vis_cols, methods_list, datasetName):
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
    fig, ax = plt.subplots(figsize=(len(vis_cols) * len(methods_list) * 0.3, 3))

    # 计算每个箱线图的宽度和位置
    n_methods = len(methods_list)
    bar_width = 0.9 / n_methods  # 每个方法的柱状图宽度
    group_width = n_methods * bar_width
    metric_positions = np.arange(len(vis_cols)) * (group_width + 0.5)  # 每组的起始位置

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
                color=sns.color_palette("Set2")[j],
                edgecolor='black',
                linewidth=0.3,  # 设置描边宽度（可调整）
                error_kw=dict(lw=0.3, capsize=1.5, capthick=0.3),  # 误差棒设置
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
    plt.savefig(f'figures/clustering_performance/{datasetName}_5_folds_klwarm_v3.png', dpi=1000, bbox_inches='tight')

    return fig


if __name__ == '__main__':
    dataset_name = 'HLCA_core'
    RBM_VAE_df = pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering.csv')
    # VAE_df = pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering.csv')
    SCVI_df = pd.read_csv(f'result/{dataset_name}/SCVI_{dataset_name}_clustering.csv')

    visualize_folds([RBM_VAE_df,  SCVI_df], ['ARI', 'AMI', 'NMI', 'HOM', 'FMI'], ['RBM_VAE', 'SCVI'], dataset_name)
