import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


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


def plot_confusion_matrix(y_true, y_pred, labels, dataset_name, fold_id, output_dir='result/', model_name=''):
    """
    绘制混淆矩阵，展示各个类别的细胞分类结果，固定为正方形，按真实标签百分比显示，不标注数字，横纵坐标轴字体大小为4。

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 类别标签（未编码的原始标签）
        dataset_name: 数据集名称
        fold_id: 交叉验证折编号
        output_dir: 输出目录
        model_name: 模型名称
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 按真实标签（行）归一化为百分比
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # 处理除零情况

    # 创建图形
    plt.figure(figsize=(3.5, 3.5))

    # 使用seaborn绘制热图，固定为正方形，不标注数字
    ax = sns.heatmap(cm_normalized,
                     annot=False,
                     cmap='Blues',
                     xticklabels=labels,
                     yticklabels=labels,
                     square=True)

    # 设置标题和标签
    plt.title(f'Confusion Matrix for {dataset_name} (Fold {fold_id})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 设置横纵坐标轴字体大小为4
    plt.tick_params(axis='x', labelsize=4)
    plt.tick_params(axis='y', labelsize=4)

    # 设置颜色条刻度标签字体大小为4
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=4)

    # 保存图像
    output_path = f'{output_dir}/{model_name}_confusion_matrix_fold{fold_id}.pdf'
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    plt.close()


def visualize_folds(performance_df_list, vis_cols, methods_list, datasetName, filename, y_start=0.3, plot_type='bar'):
    """
    Visualize the performance of different methods across folds using bar plots or box plots.

    Parameters:
    - performance_df_list: List of DataFrames, each containing performance metrics for 5 folds.
    - vis_cols: List of column names (metrics) to visualize.
    - methods_list: List of method names corresponding to performance_df_list.
    - datasetName: Name of the dataset.
    - filename: Output filename (without extension).
    - y_start: Minimum value of y-axis.
    - plot_type: 'bar' or 'box', type of plot to generate.
    """
    if len(performance_df_list) != len(methods_list):
        raise ValueError("Length of performance_df_list must match length of methods_list.")

    for df in performance_df_list:
        for col in vis_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in one of the DataFrames.")

    n_methods = len(methods_list)
    n_metrics = len(vis_cols)
    group_width = 0.8  # 控制每个 metric 的总宽度
    bar_width = group_width / n_methods

    fig, ax = plt.subplots(figsize=(n_metrics * n_methods * 0.4, 2.5))

    palette = sns.color_palette("Set2", n_methods)

    if plot_type == 'bar':
        metric_positions = np.arange(n_metrics)
        for i, metric in enumerate(vis_cols):
            for j, (method, df) in enumerate(zip(methods_list, performance_df_list)):
                metric_data = df[metric].dropna().values
                mean = np.mean(metric_data)
                std = np.std(metric_data, ddof=1)
                pos = metric_positions[i] - group_width/2 + j * bar_width + bar_width/2
                ax.bar(pos, mean,
                       yerr=std,
                       width=bar_width * 0.9,
                       color=palette[j],
                       edgecolor='black',
                       linewidth=0.3,
                       error_kw=dict(lw=0.3, capsize=1, capthick=0.3),
                       label=method if i == 0 else None)
        ax.set_xticks(metric_positions)
        ax.set_xticklabels(vis_cols)

    elif plot_type == 'box':
        all_data = []
        for metric in vis_cols:
            for method, df in zip(methods_list, performance_df_list):
                for value in df[metric].dropna().values:
                    all_data.append({
                        'Metric': metric,
                        'Method': method,
                        'Value': value
                    })
        import pandas as pd
        plot_df = pd.DataFrame(all_data)
        sns.boxplot(data=plot_df,
                    x='Metric',
                    y='Value',
                    hue='Method',
                    palette=palette,
                    ax=ax,
                    linewidth=0.5,
                    fliersize=1)

    else:
        raise ValueError("plot_type must be either 'bar' or 'box'.")

    ax.set_ylabel("Performance")
    ax.set_ylim(bottom=y_start)
    ax.set_title("Performance Across Folds")
    ax.legend(title="Methods", bbox_to_anchor=(1.02, 1), loc='upper left')

    os.makedirs('figures/clustering_performance/', exist_ok=True)
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


#   雷达图
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


