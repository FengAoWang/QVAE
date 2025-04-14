import torch
import time
import os
import numpy as np
import pandas as pd
import random  ### MODIFIED: 导入 random 模块
import logging  ### MODIFIED: 导入 logging 模块
from sklearn import metrics
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import copy
import torch.nn as nn


class scDataset(Dataset):
    def __init__(self, anndata_info):
        self.rna_tensor = torch.tensor(anndata_info, dtype=torch.float32)

    def __len__(self):
        return self.rna_tensor.shape[0]

    def __getitem__(self, idx):
        return self.rna_tensor[idx, :]


class Trainer:
    def __init__(self, model, training_config, training_data, test_data):
        self.model = model
        self.config = training_config
        self.training_data = training_data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_dir = training_config['output']
        os.makedirs(self.output_dir, exist_ok=True)
        # 检测模型是否包含RBM组件
        self.has_rbm = hasattr(model, 'rbm')  # 关键修改：动态判断模型类型

        # 统一设置随机种子  ### MODIFIED
        seed = training_config.get("random_seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 配置日志记录，根据配置决定是否保存日志  ### MODIFIED
        # 配置日志记录
        self.save_log = training_config.get("save_log", False)
        if self.save_log:
            log_file = os.path.join(self.output_dir, "train.log")
            logging.basicConfig(filename=log_file, level=logging.INFO,
                                format="%(asctime)s - %(levelname)s - %(message)s")
        else:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        self._log(f"Trainer initialized with seed {seed}. Model type: {'DVAE_RBM' if self.has_rbm else 'VAE'}")

        # 动态构建优化器参数组
        optimizer_params = []
        optimizer_params.append({'params': self.model.encoder.parameters()})
        optimizer_params.append({'params': self.model.decoder.parameters()})
        if self.has_rbm:
            optimizer_params.append({'params': self.model.rbm.parameters(), 'lr': training_config.get("rbm_lr", 1e-4)})

        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            optimizer_params,
            lr=training_config["lr"]
        )

    def _log(self, message):
        print(message)
        logging.info(message)

    # 修改 K 折交叉验证部分，将每一折信息封装到列表中  ### MODIFIED
    def create_kfold_splits(self, adata: sc.AnnData):
        indices = adata.obs.index.tolist()
        kf = KFold(n_splits=5, shuffle=True, random_state=512)
        splits = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            train_mask = adata.obs.index.isin([indices[i] for i in train_idx])
            val_mask = adata.obs.index.isin([indices[i] for i in val_idx])
            splits.append({'fold': fold, 'train': train_mask, 'valid': val_mask})
        return splits

    def train(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.config["num_epochs"]

        splits = self.create_kfold_splits(self.training_data)
        num_folds = len(splits)

        # 保存初始模型和优化器状态
        initial_model_state = copy.deepcopy(self.model.state_dict())
        initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

        for split in splits:
            fold_idx = split['fold']
            # 重置模型和优化器状态
            self.model.load_state_dict(initial_model_state)
            self.optimizer.load_state_dict(initial_optimizer_state)
            self._reset_bn()  # 重置BatchNorm统计量

            train_data = self.training_data[split['train'], :].copy()
            valid_data = self.training_data[split['valid'], :].copy()

            self._log(f"Training fold {fold_idx + 1}/{num_folds}")
            self._train_fold(train_data, valid_data, fold_idx, num_epochs)

        if self.save_log:
            self.finalize_results()

    def _reset_bn(self):
        """重置所有BatchNorm层的运行统计量"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.reset_running_stats()

    def _train_fold(self, train_data, valid_data, fold_idx, num_epochs):
        train_data_array = train_data.X.toarray() if hasattr(train_data.X, 'toarray') else train_data.X
        train_dataset = scDataset(train_data_array)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
        # 初始化指标记录
        train_elbo = []
        train_recon = []
        train_kl = []
        val_elbo = []
        val_recon = []
        val_kl = []
        metrics_dict = {
            'ARIs': [], 'AMIs': [], 'NMIs': [], 'HOMs': []
        }
        # 早停参数
        best_metric = -float('inf') if not self.has_rbm else float('inf')  # 关键修改：根据模型类型初始化
        no_improve_epochs = 0
        patience = 15  ### MODIFIED

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            self.model.train()
            total_elbo, total_recon, total_kl = 0, 0, 0

            for batch_idx, x in enumerate(train_loader):
                x = x.to(self.device)
                self.optimizer.zero_grad()

                elbo, recon_loss, kl_loss, z, zeta = self.model(x)
                loss = -elbo
                loss.backward()

                # 条件执行RBM梯度计算
                if self.has_rbm:
                    rbm_grads = self.model.rbm.compute_gradients(
                        z.detach(),
                        num_negative_samples=self.config.get('num_negative_samples', 64),
                        gibbs_steps=1
                    )
                    with torch.no_grad():
                        self.model.rbm.h.grad = rbm_grads['h']
                        self.model.rbm.W.grad = rbm_grads['W']

                self.optimizer.step()

                total_elbo += elbo.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            avg_elbo = total_elbo / len(train_loader)
            avg_recon = total_recon / len(train_loader)
            avg_kl = total_kl / len(train_loader)

            train_elbo.append(avg_elbo)
            train_recon.append(avg_recon)
            train_kl.append(avg_kl)

            epoch_time = time.time() - start_time
            self._log(f"Fold {fold_idx}, Train-Epoch [{epoch}/{num_epochs}], Time: {epoch_time:.2f}s, " +
                      f"ELBO: {avg_elbo:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")

            # 每个 epoch 均进行验证，监控 avg_recon 以便早停  ### MODIFIED
            val_results = self.val(valid_data)
            current_metric = val_results['elbo'] if not self.has_rbm else val_results['recon']
            val_elbo.append(val_results['elbo'])
            val_recon.append(val_results['recon'])
            val_kl.append(val_results['kl'])
            metrics_dict['ARIs'].append(val_results['ARI'])
            metrics_dict['AMIs'].append(val_results['AMI'])
            metrics_dict['NMIs'].append(val_results['NMI'])
            metrics_dict['HOMs'].append(val_results['HOM'])

            # 早停判定：若验证重构 loss 未改进超过 patience 次则提前停止  ### MODIFIED
            if not self.has_rbm:
                # VAE 监控 ELBO（越大越好）
                improve_condition = current_metric > best_metric
            else:
                # DVAE_RBM 监控 Recon（越小越好）
                improve_condition = current_metric < best_metric

            if improve_condition:
                best_metric = current_metric
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                self._log(f"Early stopping triggered at epoch {epoch}. No improvement for {patience} epochs.")
                break

        # 保存当前折的训练结果
        self._save_results(fold_idx, train_elbo, train_recon, train_kl, val_elbo, val_recon, val_kl, metrics_dict)

    def val(self, valid_data):
        valid_data_array = valid_data.X.toarray() if hasattr(valid_data.X, 'toarray') else valid_data.X
        valid_dataset = scDataset(valid_data_array)
        valid_loader = DataLoader(valid_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=4)

        self.model.eval()
        val_total_elbo, val_total_recon, val_total_kl = 0, 0, 0
        latent_reps = []

        with torch.no_grad():
            for x in valid_loader:
                x = x.to(self.device)
                elbo, recon_loss, kl_loss, z, zeta = self.model(x)
                latent_reps.append(zeta.cpu().numpy())

                val_total_elbo += elbo.item()
                val_total_recon += recon_loss.item()
                val_total_kl += kl_loss.item()

            val_avg_elbo = val_total_elbo / len(valid_loader)
            val_avg_recon = val_total_recon / len(valid_loader)
            val_avg_kl = val_total_kl / len(valid_loader)

        self._log(f"Validation-Epoch, ELBO: {val_avg_elbo:.4f}, Recon: {val_avg_recon:.4f}, KL: {val_avg_kl:.4f}")

        qvae_hidden_reps = np.concatenate(latent_reps, axis=0)
        valid_data.obsm['qvae_reps'] = qvae_hidden_reps

        val_results = self._calculate_clustering_metrics(valid_data)
        val_results.update({
            'elbo': val_avg_elbo,
            'recon': val_avg_recon,
            'kl': val_avg_kl
        })

        return val_results

    def _calculate_clustering_metrics(self, validation_data):
        if 'cell_type' not in validation_data.obs:
            self._log("Warning: 'cell_type' not found in validation_data.obs. Skipping clustering metrics.")
            return {'ARI': 0, 'AMI': 0, 'NMI': 0, 'HOM': 0}

        sc.pp.neighbors(validation_data, n_neighbors=10, use_rep='qvae_reps')
        sc.tl.leiden(validation_data)

        ARI = metrics.adjusted_rand_score(validation_data.obs['cell_type'], validation_data.obs['leiden'])
        AMI = metrics.adjusted_mutual_info_score(validation_data.obs['cell_type'], validation_data.obs['leiden'])
        NMI = metrics.normalized_mutual_info_score(validation_data.obs['cell_type'], validation_data.obs['leiden'])
        HOM = metrics.homogeneity_score(validation_data.obs['cell_type'], validation_data.obs['leiden'])

        self._log(f'Clustering metrics => ARI: {ARI:.4f}, AMI: {AMI:.4f}, NMI: {NMI:.4f}, HOM: {HOM:.4f}')
        return {'ARI': ARI, 'AMI': AMI, 'NMI': NMI, 'HOM': HOM}

    # 修改 _save_results，单独画图，并保存 CSV 文件  ### MODIFIED
    def _save_results(self, fold_idx, train_elbo, train_recon, train_kl, val_elbo, val_recon, val_kl, metrics):
        # 创建结果 DataFrame
        epochs = np.arange(1, len(train_elbo) + 1)
        # 保持验证数据的 epoch 记录
        val_epochs = np.arange(1, len(val_elbo) + 1)

        train_df = pd.DataFrame({
            'epoch': epochs,
            'elbo': train_elbo,
            'recon': train_recon,
            'kl': train_kl
        })

        val_df = pd.DataFrame({
            'epoch': val_epochs,
            'elbo': val_elbo,
            'recon': val_recon,
            'kl': val_kl,
            'ARI': metrics['ARIs'],
            'AMI': metrics['AMIs'],
            'NMI': metrics['NMIs'],
            'HOM': metrics['HOMs']
        })

        fold_dir = os.path.join(self.output_dir, f'fold_{fold_idx + 1}')  ### MODIFIED
        os.makedirs(fold_dir, exist_ok=True)

        train_df.to_csv(os.path.join(fold_dir, 'train_results.csv'), index=False)
        val_df.to_csv(os.path.join(fold_dir, 'val_results.csv'), index=False)

        # 调用画图方法
        self._plot_results(fold_dir, train_df, val_df)

    # 修改 _plot_results，训练和验证各指标分别画图，保存为单个文件  ### MODIFIED
    def _plot_results(self, fold_dir, train_df, val_df):
        # 分别绘制训练指标
        for metric in ['elbo', 'recon', 'kl']:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(train_df['epoch'], train_df[metric], label=f"Train {metric}")
            ax.set_title(f"Train {metric.upper()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.upper())
            ax.legend()
            plt.savefig(os.path.join(fold_dir, f"train_{metric}.png"), dpi=300)
            plt.close()

        # 分别绘制验证指标
        for metric in ['elbo', 'recon', 'kl']:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(val_df['epoch'], val_df[metric], label=f"Val {metric}")
            ax.set_title(f"Val {metric.upper()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.upper())
            ax.legend()
            plt.savefig(os.path.join(fold_dir, f"val_{metric}.png"), dpi=300)
            plt.close()

        # 聚类指标绘图（多指标在同一图）
        fig, ax = plt.subplots(figsize=(6, 6))
        for metric in ['ARI', 'AMI', 'NMI', 'HOM']:
            ax.plot(val_df['epoch'], val_df[metric], label=metric)
        ax.set_title("Clustering Metrics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend()
        plt.savefig(os.path.join(fold_dir, "clustering_metrics.png"), dpi=300)
        plt.close()

    # 聚合所有 fold 的 val_results 并计算均值，生成 mean_val_result.csv，然后绘图  ### MODIFIED
    def finalize_results(self):
        self._log("Finalizing results: aggregating validation results from all folds.")
        val_results_list = []
        max_common_epoch = None  # 新增：记录所有fold共有的最大epoch数
        # 第一次遍历：确定所有fold共有的最小epoch数
        min_epochs = []
        for fold in os.listdir(self.output_dir):
            fold_dir = os.path.join(self.output_dir, fold)
            if os.path.isdir(fold_dir) and fold.startswith("fold_"):
                val_file = os.path.join(fold_dir, "val_results.csv")
                if os.path.exists(val_file):
                    df = pd.read_csv(val_file)
                    min_epochs.append(df['epoch'].max())
        if min_epochs:
            max_common_epoch = min(min_epochs)  # 取所有fold的最小最大epoch数
        else:
            return

        # 第二次遍历：截断数据到共有的max_common_epoch
        for fold in os.listdir(self.output_dir):
            fold_dir = os.path.join(self.output_dir, fold)
            if os.path.isdir(fold_dir) and fold.startswith("fold_"):
                val_file = os.path.join(fold_dir, "val_results.csv")
                if os.path.exists(val_file):
                    df = pd.read_csv(val_file)
                    df = df[df['epoch'] <= max_common_epoch]  # 截断到共有的epoch数
                    df['fold'] = fold
                    val_results_list.append(df)

        if not val_results_list:
            self._log("No val_results.csv found in any fold folder.")
            return

        # 将所有结果合并，按 epoch 进行分组平均（假设各 fold 的 epoch 数一致）
        all_val_df = pd.concat(val_results_list, ignore_index=True)
        mean_val_df = all_val_df.groupby('epoch').mean().reset_index()
        mean_val_file = os.path.join(self.output_dir, "mean_val_result.csv")
        mean_val_df.to_csv(mean_val_file, index=False)
        self._log(f"Mean validation results saved to {mean_val_file}")

        # 分别画图：ELBO, Recon, KL 为单图；ARI, AMI, NMI, HOM 在一图中
        for metric in ['elbo', 'recon', 'kl']:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(mean_val_df['epoch'], mean_val_df[metric], label=metric.upper())
            ax.set_title(f"Mean Validation {metric.upper()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.upper())
            ax.legend()
            plt.savefig(os.path.join(self.output_dir, f"mean_val_{metric}.png"), dpi=300)
            plt.close()

        fig, ax = plt.subplots(figsize=(6, 6))
        for metric in ['ARI', 'AMI', 'NMI', 'HOM']:
            ax.plot(mean_val_df['epoch'], mean_val_df[metric], label=metric)
        ax.set_title("Mean Clustering Metrics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend()
        plt.savefig(os.path.join(self.output_dir, "mean_clustering_metrics.png"), dpi=300)
        plt.close()
        self._log("Aggregation and plotting of mean validation results completed.")
