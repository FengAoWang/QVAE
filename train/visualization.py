import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from model.visualization_func import *


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
    # dataset_name = 'Lung_atlas'
    # RBM_VAE_df = pd.read_csv(
    #     f'result/{dataset_name}/RBM_VAE_{dataset_name}_clustering_latentDim256_layernorm_batchSize256_multiLayers_weight_decay.csv')
    # RBM_VAE_df['Bio conservation'] = RBM_VAE_df[col_use].mean(axis=1, skipna=True)
    #
    # # VAE_df = pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim128_weight_decay.csv')
    # # VAE_df['Bio conservation'] = VAE_df[col_use].mean(axis=1, skipna=True)
    #
    # SCVI_df = pd.read_csv(f'result/{dataset_name}/SCVI_{dataset_name}_clustering.csv')
    # SCVI_df['Bio conservation'] = SCVI_df[col_use].mean(axis=1, skipna=True)
    #
    # # tr_VAE_df = pd.read_csv(f'result/{dataset_name}/trVAE_{dataset_name}_clustering.csv')
    # # LDVAE_df = pd.read_csv(f'result/{dataset_name}/LDVAE_{dataset_name}_clustering.csv')
    # # LDVAE_df['Bio conservation'] = LDVAE_df[col_use].mean(axis=1, skipna=True)
    # #
    # # AUTOZI_df = pd.read_csv(f'result/{dataset_name}/AUTOZI_{dataset_name}_clustering.csv')
    # # AUTOZI_df['Bio conservation'] = AUTOZI_df[col_use].mean(axis=1, skipna=True)
    #
    #
    # # visualize_folds([RBM_VAE_df,  RBM_VAE_df2, RBM_VAE_df3, SCVI_df], ['ARI', 'AMI', 'NMI', 'HOM', 'FMI'], ['RBM_VAE', 'RBM_VAE_wd_layer_bs2048', 'RBM_VAE_wd', 'SCVI'], dataset_name, 'bio_conservation')
    # #
    # visualize_folds([RBM_VAE_df, SCVI_df,
    #                  # LDVAE_df, AUTOZI_df, VAE_df
    #                  ],
    #                 ['leiden_ARI', 'leiden_AMI', 'leiden_NMI', 'leiden_FMI', 'Bio conservation'],
    #                 ['RBM_VAE', 'SCVI'
    #                     # , 'LDVAE', 'AUTOZI', 'VAE'
    #                  ],
    #                 dataset_name,
    #                 'bio_conservation_bm')

    # #benchmark
    col_use = ['acc', 'pre', 'recall', 'f1']
    dataset_name = 'BMMC_multiome'
    RBM_VAE_df = pd.read_csv(
        f'result/{dataset_name}/RBM_VAE_{dataset_name}_cell_classifying_latentDim256_layernorm_batchSize256_multiLayers_weight_decay.csv')

    # VAE_df = pd.read_csv(f'result/{dataset_name}/VAE_{dataset_name}_clustering_latentDim128_weight_decay.csv')
    # VAE_df['Bio conservation'] = VAE_df[col_use].mean(axis=1, skipna=True)

    SCVI_df = pd.read_csv(f'result/{dataset_name}/SCVI_{dataset_name}_cell_classify.csv')

    # tr_VAE_df = pd.read_csv(f'result/{dataset_name}/trVAE_{dataset_name}_clustering.csv')
    # LDVAE_df = pd.read_csv(f'result/{dataset_name}/LDVAE_{dataset_name}_clustering.csv')
    # LDVAE_df['Bio conservation'] = LDVAE_df[col_use].mean(axis=1, skipna=True)
    #
    # AUTOZI_df = pd.read_csv(f'result/{dataset_name}/AUTOZI_{dataset_name}_clustering.csv')
    # AUTOZI_df['Bio conservation'] = AUTOZI_df[col_use].mean(axis=1, skipna=True)


    # visualize_folds([RBM_VAE_df,  RBM_VAE_df2, RBM_VAE_df3, SCVI_df], ['ARI', 'AMI', 'NMI', 'HOM', 'FMI'], ['RBM_VAE', 'RBM_VAE_wd_layer_bs2048', 'RBM_VAE_wd', 'SCVI'], dataset_name, 'bio_conservation')
    #
    visualize_folds([RBM_VAE_df, SCVI_df,
                     # LDVAE_df, AUTOZI_df, VAE_df
                     ],
                    ['acc', 'pre', 'recall', 'f1'],
                    ['RBM_VAE', 'SCVI'
                        # , 'LDVAE', 'AUTOZI', 'VAE'
                     ],
                    dataset_name,
                    'cell_classification_bm',
                    y_start=0.7)


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


    # import pickle
    #
    # # Path to your .pkl file
    # gibbs_path = '/home/wfa/project/QVAE/model/figures/clustering_performance/gibbs_val_elbo.pkl'
    # CIM_path = '/home/wfa/project/QVAE/model/figures/clustering_performance/cim_val_elbo.pkl'
    #
    # # Open and load the pickle file
    # with open(gibbs_path, 'rb') as file:
    #     gibbs_data = pickle.load(file)
    #
    # with open(CIM_path, 'rb') as file:
    #     CIM_data = pickle.load(file)
    #
    # # Inspect the contents
    #
    # plot_loss_curves([gibbs_data['fold1'], CIM_data['fold1']], ['Gibbs', 'CIM'],)
    # plot_loss_curves([gibbs_data['fold1'], CIM_data['fold1']], ['Gibbs', 'CIM'], start_step=100, output_file='step100.pdf')

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
