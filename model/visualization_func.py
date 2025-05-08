import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# def

def visualize_folds(performance_df_list, vis_cols, methods_list, datasetName, filename, y_start=0.3):
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
                color=sns.color_palette("Set2")[j],
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
    ax.set_ylim(bottom=y_start)

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
    plt.figure(figsize=(2.5, 2))  # Slightly larger for clarity with multiple methods

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
                         label=f'{col}',
                         markersize=2)

            if len(methodsList) > 1:
                plt.plot(summary_df[x_dim_name], summary_df[col]['mean'],
                         marker='o', color=colors[color_idx],
                         label=f'{method} {col}',
                         markersize=2)
            # Plot error band with matching color
            plt.fill_between(summary_df[x_dim_name],
                             summary_df[col]['mean'] - summary_df[col]['std'],
                             summary_df[col]['mean'] + summary_df[col]['std'],
                             color=colors[color_idx], alpha=0.1)
            color_idx += 1  # Move to next color for next method-metric pair

    # Customize plot
    plt.xlabel(x_dim_name)
    plt.ylabel('Value')
    plt.title(f'{datasetName}')
    plt.xticks(latent_dims, latent_dims, rotation=40)  # Uniform x-ticks
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(f'figures/clustering_latent/{datasetName}_{x_dim_name}_{figure_name}.pdf', dpi=1000,
                bbox_inches='tight')


def plot_radar_chart(df,
                     title="Radar Chart",
                     figsize=(2, 2),
                     colormap='husl',
                     linestyle='solid',
                     smooth=True,
                     save_path=None,
                     r_range=None,
                     num_radial_ticks=3,
                     interp_points=200,
                     spline_degree=2
                     ):
    import matplotlib.cm as cm
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns  # Ensure seaborn is imported for color palette

    """
    Plot a radar chart from a DataFrame where rows are methods and columns are datasets.

    Parameters:
    df (pd.DataFrame): DataFrame with methods as index and datasets as columns
    title (str): Title of the radar chart
    figsize (tuple): Figure size as (width, height)
    colormap (str): Matplotlib colormap name (e.g., 'tab10', 'viridis', 'Set2', 'plasma').
                    See matplotlib.cm for available colormaps.
    linestyle (str): Line style for plotting (e.g., 'solid', 'dashed', 'dotted')
    smooth (bool): If True, interpolate data for smoother curves
    save_path (str): Path to save the figure (optional)
    r_range (tuple): Range of radial axis (r_min, r_max). If None, autoscaled.

    Raises:
    ValueError: If colormap is not a valid Matplotlib colormap
    """

    # Number of variables (datasets)
    num_vars = len(df.columns)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Initialize the figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Get colors from colormap
    colors = sns.color_palette(colormap, len(df.index))  # Use number of methods for colors

    # Plot each method
    for idx, method in enumerate(df.index):
        values = df.loc[method].values.tolist()
        values += values[:1]  # Complete the loop

        if smooth:
            # Interpolate for smoother curves
            from scipy.interpolate import make_interp_spline
            theta = np.array(angles)
            vals = np.array(values)
            spline = make_interp_spline(theta[:-1], vals[:-1], k=spline_degree)
            smooth_theta = np.linspace(theta[0], theta[-2], interp_points)
            smooth_vals = spline(smooth_theta)
            smooth_vals = np.append(smooth_vals, smooth_vals[0])  # Close the loop
            smooth_theta = np.append(smooth_theta, smooth_theta[0])
            ax.plot(smooth_theta, smooth_vals, linewidth=1, linestyle=linestyle, label=method, color=colors[idx], antialiased=True)
            ax.fill(smooth_theta, smooth_vals, alpha=0.05, color=colors[idx], antialiased=True)
            # Plot markers at original data points
            ax.plot(angles, values, 'o', markersize=2, color=colors[idx], alpha=0.9)
        else:
            ax.plot(angles, values, linewidth=1.5, linestyle=linestyle, label=method, color=colors[idx], antialiased=True)
            ax.fill(angles, values, alpha=0.2, color=colors[idx], antialiased=True)
            # Plot markers at data points
            ax.plot(angles, values, 'o', markersize=5, color=colors[idx], alpha=0.7)

    # Set radial range if specified
    if r_range is not None:
        r_min, r_max = r_range
        ax.set_ylim(r_min, r_max)

    # Remove radial grid (circular background)
    # ax.set_yticks([])  # Disable radial ticks and grid lines
    # Set radial ticks
    radial_ticks = np.linspace(r_min, r_max, num_radial_ticks)
    ax.set_yticks(radial_ticks)  # Set specified number of radial ticks
    ax.set_yticklabels([f"{tick:.2f}" for tick in radial_ticks], fontsize=8)  # Add labels with formatting
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(df.columns)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Set title
    plt.title(title)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=1000)
    plt.close()


def plot_score_rank_circles(
    df_scores: pd.DataFrame,
    rank_ascending: bool = False,
    cmap_ranks: str = 'viridis_r',
    max_circle_area: float = 300,
    score_range: tuple = None,
    figsize: tuple = (7, 2.5),  # Fixed default size, adjust as needed
    title: str = 'Method Performance Comparison',
    circle_edge_color: str = 'grey',
    output_file: str = None
):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import numpy as np

    # --- 1. Data Preparation and Ranking ---
    if not isinstance(df_scores, pd.DataFrame):
        raise TypeError("Input 'df_scores' must be a pandas DataFrame.")

    methods = df_scores.index.tolist()
    metrics = df_scores.columns.tolist()
    n_methods = len(methods)
    n_metrics = len(metrics)

    if n_methods == 0 or n_metrics == 0:
        print("Input DataFrame is empty. Cannot generate plot.")
        return None, None

    # Calculate Ranks
    df_ranks = df_scores.rank(axis=0, ascending=rank_ascending, method='min').astype(int)

    # --- 2. Define Plot Parameters ---
    # Determine score range for sizing
    if score_range is None:
        min_score_overall = df_scores.min(skipna=True).min(skipna=True)
        max_score_overall = df_scores.max(skipna=True).max(skipna=True)
        if pd.isna(min_score_overall) or pd.isna(max_score_overall):
            print("Warning: Could not determine score range from data (all NaN?). Using [0, 1].")
            min_score_overall, max_score_overall = 0, 1
    else:
        min_score_overall, max_score_overall = score_range

    # Size scaling function
    def score_to_area(score):
        if pd.isna(score):
            return 0
        if max_score_overall == min_score_overall:
            return max_circle_area / 2
        clamped_score = np.clip(score, min_score_overall, max_score_overall)
        normalized_score = (clamped_score - min_score_overall) / (max_score_overall - min_score_overall)
        return max(0, normalized_score * max_circle_area) + 1e-6

    # Color mapping for ranks
    try:
        cmap_obj = plt.get_cmap(cmap_ranks)
    except ValueError:
        print(f"Warning: Colormap '{cmap_ranks}' not found. Using 'viridis_r'.")
        cmap_obj = plt.get_cmap('viridis_r')
    norm_ranks = mcolors.Normalize(vmin=1, vmax=n_methods)
    mapper_ranks = cm.ScalarMappable(norm=norm_ranks, cmap=cmap_obj)

    # --- 3. Create the Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # Iterate through data to plot circles
    for r_idx, method in enumerate(methods):
        for c_idx, metric in enumerate(metrics):
            rank = df_ranks.loc[method, metric]
            score = df_scores.loc[method, metric]

            if pd.isna(rank) or pd.isna(score):
                ax.scatter(c_idx, r_idx, s=max_circle_area*0.05, marker='x', color='lightgrey', alpha=0.7)
                continue

            circle_area = score_to_area(score)
            rank_color_rgba = mapper_ranks.to_rgba(rank)

            if circle_area > 1e-6:
                ax.scatter(c_idx, r_idx,
                           s=circle_area,
                           c=[rank_color_rgba],
                           alpha=0.85,
                           edgecolors=circle_edge_color if circle_edge_color else 'none',
                           linewidths=0.5 if circle_edge_color else 0)

    # --- 4. Configure Axes ---
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_methods))
    ax.set_xticklabels(metrics, rotation=45)
    ax.set_yticklabels(methods)

    # Ensure consistent row height by fixing Y limits
    ax.set_xlim(-0.5, n_metrics - 0.5)
    ax.set_ylim(-0.5, n_methods - 0.5)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', length=0, pad=2)
    ax.tick_params(axis='y', length=0, pad=2)

    # Add grid
    ax.grid(True, which='major', axis='both', linestyle=':', color='lightgray', alpha=0.6)
    ax.set_axisbelow(True)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- 5. Add Legends ---
    cbar = fig.colorbar(mapper_ranks, ax=ax, orientation='vertical', shrink=0.6, aspect=15,
                        anchor=(0.0, 1.0), location='right', pad=0.05)
    cbar.set_label(f"Rank ({'Lower' if rank_ascending else 'Higher'} Score is Better Rank 1)",
                   rotation=270, labelpad=15, fontsize=9)
    if n_methods <= 10:
        tick_positions = np.arange(1, n_methods + 1)
    else:
        tick_positions = np.linspace(1, n_methods, 5).astype(int)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_positions)
    if not rank_ascending:
        cbar.ax.invert_yaxis()
    cbar.ax.tick_params(labelsize=6)

    # Size legend
    n_legend_points = 3
    legend_scores_raw = np.linspace(min_score_overall, max_score_overall, n_legend_points)
    if min_score_overall == max_score_overall:
        legend_scores_raw = [min_score_overall]
        n_legend_points = 1
    legend_scores = [s for s in legend_scores_raw if not pd.isna(s)]
    legend_handles = [plt.scatter([], [], s=max(5, score_to_area(s)), c='grey', alpha=0.8) for s in legend_scores]
    legend_labels = [f"{int(s)}" if abs(s - int(s)) < 1e-9 else f"{s:.2g}" for s in legend_scores]

    size_legend = ax.legend(handles=legend_handles,
                            labels=legend_labels,
                            title="Score (Size)",
                            loc='lower right',
                            bbox_to_anchor=(1.35, 0.3),
                            frameon=False,
                            labelspacing=1.8,
                            borderpad=1,
                            fontsize=6,
                            title_fontsize=10)
    ax.add_artist(size_legend)

    # --- 6. Final Touches ---
    if title:
        plt.suptitle(title, y=0.98)  # Fixed title position

    # Fixed layout to ensure consistent margins
    fig.tight_layout(rect=[0, 0, 0.85, 0.95])

    # --- 7. Output ---
    if output_file:
        try:
            fig.savefig(output_file, bbox_inches='tight', dpi=1000)
            print(f"Plot saved to {output_file}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        plt.show()

    return fig, ax


def plot_loss_curves(loss_lists, method_names, start_step=0, title="Loss vs. Training Steps",
                     xlabel="Training Steps", ylabel="Loss", output_file="loss_plot.pdf",
                     color_palette="Set1", line_styles=['-', '--']):
    """
    Visualize loss curves for multiple methods over training steps with customizable colors and line styles.

    Args:
        loss_lists (list[list[float]]): List of lists, where each inner list contains loss values for a method.
        method_names (list[str]): List of method names corresponding to each loss list.
        start_step (int, optional): Starting step for plotting. Defaults to 0.
        title (str, optional): Plot title. Defaults to "Loss vs. Training Steps".
        xlabel (str, optional): X-axis label. Defaults to "Training Steps".
        ylabel (str, optional): Y-axis label. Defaults to "Loss".
        output_file (str, optional): File name for saving the plot. Defaults to "loss_plot.pdf".
        color_palette (str, optional): Name of the Matplotlib/Seaborn color palette. Defaults to "tab10".
        line_styles (list[str], optional): List of line styles (e.g., ['-', '--', ':'']). Defaults to None (all solid).
    """
    plt.figure(figsize=(2.5, 2.5))

    # Ensure start_step is non-negative
    start_step = max(0, start_step)

    # Get colors from the specified palette
    colors = sns.color_palette(color_palette, n_colors=len(method_names))

    # Set default line styles if none provided
    if line_styles is None:
        line_styles = ['-'] * len(method_names)
    else:
        # Ensure line_styles length matches method_names; cycle through if shorter
        line_styles = (line_styles * (len(method_names) // len(line_styles) + 1))[:len(method_names)]

    # Plot each method's loss curve
    for losses, method_name, color, style in zip(loss_lists, method_names, colors, line_styles):
        # Truncate losses and steps based on start_step
        steps = np.arange(start_step, start_step + len(losses[start_step:]))
        plt.plot(steps, losses[start_step:], label=method_name, color=color, linestyle=style, linewidth=1)

    # Customize the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    # Save the plot to a file
    plt.savefig(output_file, bbox_inches='tight', dpi=1000)
    plt.close()


if __name__ == '__main__':


    # from dataset_param_dict import dataset_params
    #
    # # overall 雷达图
    # df_dict = {}
    # for dataset_name in dataset_params.keys():
    #     RBM_VAE_df = pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim256_layernorm_batchSize128_multiLayers_weight_decay.csv')
    #     VAE_df = pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim128_weight_decay.csv')
    #     SCVI_df = pd.read_csv(f'result/{dataset_name}/SCVI_{dataset_name}_clustering.csv')
    #     LDVAE_df = pd.read_csv(f'result/{dataset_name}/LDVAE_{dataset_name}_clustering.csv')
    #     AUTOZI_df = pd.read_csv(f'result/{dataset_name}/AUTOZI_{dataset_name}_clustering.csv')
    #
    #     RBM_average_bio = RBM_VAE_df[['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI']].mean().mean()
    #     VAE_average_bio = VAE_df[['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI']].mean().mean()
    #     SCVI_average_bio = SCVI_df[['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI']].mean().mean()
    #     LDVAE_average_bio = LDVAE_df[['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI']].mean().mean()
    #     AUTOZI_average_bio = AUTOZI_df[['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI']].mean().mean()
    #
    #     df_dict[dataset_name] = [RBM_average_bio, VAE_average_bio, SCVI_average_bio, LDVAE_average_bio, AUTOZI_average_bio]
    #
    # df = pd.DataFrame(df_dict, index=['RBM-VAE', 'VAE', 'SCVI', 'LDVAE', 'AUTOZI'])
    # plot_radar_chart(df, title="Model Performance Comparison", save_path='overall_bio.pdf', r_range=[0.5, 0.75])
    #
    # for dataset_name in dataset_params.keys():
    #     RBM_VAE_df = pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim256_layernorm_batchSize128_multiLayers_weight_decay.csv')
    #     VAE_df = pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim128_weight_decay.csv')
    #     SCVI_df = pd.read_csv(f'result/{dataset_name}/SCVI_{dataset_name}_clustering.csv')
    #     LDVAE_df = pd.read_csv(f'result/{dataset_name}/LDVAE_{dataset_name}_clustering.csv')
    #     AUTOZI_df = pd.read_csv(f'result/{dataset_name}/AUTOZI_{dataset_name}_clustering.csv')
    #
    #     RBM_average_bio = RBM_VAE_df[['Batch correction']].mean().mean()
    #     VAE_average_bio = VAE_df[['Batch correction']].mean().mean()
    #     SCVI_average_bio = SCVI_df[['Batch correction']].mean().mean()
    #     LDVAE_average_bio = LDVAE_df[['Batch correction']].mean().mean()
    #     AUTOZI_average_bio = AUTOZI_df[['Batch correction']].mean().mean()
    #
    #     df_dict[dataset_name] = [RBM_average_bio, VAE_average_bio, SCVI_average_bio, LDVAE_average_bio, AUTOZI_average_bio]
    #
    # df = pd.DataFrame(df_dict, index=['RBM-VAE', 'VAE', 'SCVI', 'LDVAE', 'AUTOZI'])
    # plot_radar_chart(df, title="Model Performance Comparison", save_path='overall_batch.pdf', r_range=[0.4, 0.75])
    #
    #
    #
    # #    heatmap
    # for dataset_name in dataset_params.keys():
    #     col_use = ['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison', 'Batch correction']
    #     batch_use = ['iLISI', 'KBET', 'Graph connectivity', 'PCR comparison', 'Batch correction']
    #
    #     RBM_VAE_df = pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim256_layernorm_batchSize128_multiLayers_weight_decay.csv')
    #     VAE_df = pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim128_weight_decay.csv')
    #     SCVI_df = pd.read_csv(f'result/{dataset_name}/SCVI_{dataset_name}_clustering.csv')
    #     LDVAE_df = pd.read_csv(f'result/{dataset_name}/LDVAE_{dataset_name}_clustering.csv')
    #     AUTOZI_df = pd.read_csv(f'result/{dataset_name}/AUTOZI_{dataset_name}_clustering.csv')
    #
    #     RBM_average_bio = RBM_VAE_df.mean()
    #     VAE_average_bio = VAE_df.mean()
    #     SCVI_average_bio = SCVI_df.mean()
    #     LDVAE_average_bio = LDVAE_df.mean()
    #     AUTOZI_average_bio = AUTOZI_df.mean()
    #     # 为每个方法添加名称
    #     RBM_average_bio.name = f'RBM-VAE'
    #     VAE_average_bio.name = f'VAE'
    #     SCVI_average_bio.name = f'SCVI'
    #     LDVAE_average_bio.name = f'LDVAE'
    #     AUTOZI_average_bio.name = f'AUTOZI'
    #
    #     # 合并为一个 DataFrame
    #     all_df = pd.concat([RBM_average_bio, VAE_average_bio, SCVI_average_bio, LDVAE_average_bio, AUTOZI_average_bio],
    #                        axis=1).T
    #     bio_all_df = all_df[col_use]
    #     bio_all_df['Bio conservation'] = bio_all_df[col_use].mean(axis=1, skipna=True)
    #     batch_all_df = all_df[batch_use]
    #
    #     # 打印当前数据集的 DataFrame
    #     print(f"\nDataFrame for {dataset_name}:")
    #     # Customized usage
    #     plot_score_rank_circles(
    #         bio_all_df,
    #         rank_ascending=False,  # Higher score = Rank 1
    #         cmap_ranks='Reds_r',  # Different colormap
    #         score_range=(0, 1.0),
    #         title="Model Benchmark Results",
    #         output_file=f"{dataset_name}_benchmark_bio_plot.pdf" # Uncomment to save instead of show
    #     )
    #
    #     plot_score_rank_circles(
    #         batch_all_df,
    #         rank_ascending=False,  # Higher score = Rank 1
    #         cmap_ranks='Blues_r',  # Different colormap
    #         score_range=(0, 1.0),
    #         title="Model Benchmark Results",
    #         output_file=f"{dataset_name}_benchmark_batch_plot.pdf" # Uncomment to save instead of show
    #     )
    #
    #
    # #benchmark
    # col_use = ['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI',]
    # dataset_name = 'HLCA_core'
    # RBM_VAE_df = pd.read_csv(
    #     f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim256_layernorm_batchSize128_multiLayers_weight_decay.csv')
    # RBM_VAE_df['Bio conservation'] = RBM_VAE_df[col_use].mean(axis=1, skipna=True)
    #
    # VAE_df = pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim128_weight_decay.csv')
    # VAE_df['Bio conservation'] = VAE_df[col_use].mean(axis=1, skipna=True)
    #
    # SCVI_df = pd.read_csv(f'result/{dataset_name}/SCVI_{dataset_name}_clustering.csv')
    # SCVI_df['Bio conservation'] = SCVI_df[col_use].mean(axis=1, skipna=True)
    #
    # # tr_VAE_df = pd.read_csv(f'result/{dataset_name}/trVAE_{dataset_name}_clustering.csv')
    # LDVAE_df = pd.read_csv(f'result/{dataset_name}/LDVAE_{dataset_name}_clustering.csv')
    # LDVAE_df['Bio conservation'] = LDVAE_df[col_use].mean(axis=1, skipna=True)
    #
    # AUTOZI_df = pd.read_csv(f'result/{dataset_name}/AUTOZI_{dataset_name}_clustering.csv')
    # AUTOZI_df['Bio conservation'] = AUTOZI_df[col_use].mean(axis=1, skipna=True)
    #
    #
    # # visualize_folds([RBM_VAE_df,  RBM_VAE_df2, RBM_VAE_df3, SCVI_df], ['ARI', 'AMI', 'NMI', 'HOM', 'FMI'], ['RBM_VAE', 'RBM_VAE_wd_layer_bs2048', 'RBM_VAE_wd', 'SCVI'], dataset_name, 'bio_conservation')
    #
    # visualize_folds([RBM_VAE_df, SCVI_df, LDVAE_df, AUTOZI_df, VAE_df],
    #                 ['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI', 'Bio conservation'],
    #                 ['RBM_VAE', 'SCVI', 'LDVAE', 'AUTOZI', 'VAE'],
    #                 dataset_name,
    #                 'bio_conservation_bm')
    # visualize_folds([RBM_VAE_df, SCVI_df, LDVAE_df, AUTOZI_df, VAE_df],
    #                 ['iLISI', 'KBET', 'Graph connectivity', 'Batch correction'],
    #                 ['RBM_VAE', 'SCVI', 'LDVAE', 'AUTOZI', 'VAE'],
    #                 dataset_name,
    #                 'batch_remove_bm',
    #                 y_start=0)
    #
    #
    # CIM_df = pd.read_csv('/home/wfa/project/QVAE/model/result/pancreas/cim_result.csv')
    # gibbs_df = pd.read_csv('/home/wfa/project/QVAE/model/result/pancreas/gibbs_result.csv')
    #
    # visualize_folds([CIM_df, gibbs_df],
    #                 ['ARI', 'AMI', 'NMI', 'FMI'],
    #                 ['CIM', "Gibbs"],
    #                 'pancreas',
    #                 'CIM')
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
    # visualize_latentDims(df_list, ['ARI', 'AMI', 'NMI', 'FMI'], latent_dims, dataset_name, methods, 'latent dim', 'compare')


    import pickle

    # Path to your .pkl file
    gibbs_path = '/home/wfa/project/QVAE/model/figures/clustering_performance/gibbs_val_elbo.pkl'
    CIM_path = '/home/wfa/project/QVAE/model/figures/clustering_performance/cim_val_elbo.pkl'

    # Open and load the pickle file
    with open(gibbs_path, 'rb') as file:
        gibbs_data = pickle.load(file)

    with open(CIM_path, 'rb') as file:
        CIM_data = pickle.load(file)

    # Inspect the contents

    plot_loss_curves([gibbs_data['fold1'], CIM_data['fold1']], ['Gibbs', 'CIM'],)
    plot_loss_curves([gibbs_data['fold1'], CIM_data['fold1']], ['Gibbs', 'CIM'], start_step=100, output_file='step100.pdf')

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

    # RBM_df_list = []
    # VAE_df_list = []
    # batch_dims = [128, 256, 512, 1024, 2048]
    # methods = ['RBM-VAE']
    #
    # dataset_name = 'immune'
    # for batch_dim in batch_dims:
    #     RBM_df_list.append(pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim256_layernorm_batchSize{batch_dim}_weight_decay.csv'))
    #
    # # for latent_dim in latent_dims:
    # #     VAE_df_list.append(pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim{latent_dim}_weight_decay.csv'))
    #
    # # for latent_dim in latent_dims:
    # #     RBM_df_list2.append(pd.read_csv(f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim{latent_dim}_weight_decay.csv'))
    # df_list = [RBM_df_list]
    #
    # # for col in ['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI']:
    # visualize_latentDims(df_list, ['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI'], batch_dims, dataset_name, methods, 'batch size', 'layernorm')
