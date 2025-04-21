
dataset_params = {
    # 'pancreas': dict(
    # # dataset
    # dataset_name="pancreas",
    # batch_key="tech",
    # labels_key="celltype",
    # file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/pancreas_processed.h5ad"),

    # 'BMMC_': dict(
    #     # dataset
    #     dataset_name="BMMC",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/BMMC_processed.h5ad"
    # ),
    # 'fetal_lung': dict(
    #     # dataset
    #     dataset_name="fetal_lung",
    #     batch_key="batch",
    #     labels_key="broad_celltype",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/fetal_lung_processed.h5ad"
    # ),
    # 'immune': dict(
    #     dataset_name="immune",
    #     batch_key="batch",
    #     labels_key="final_annotation",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/immune_processed.h5ad"
    # ),
    #
    # 'BMMC_multiome': dict(
    #     dataset_name="BMMC_multiome",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/project/single_cell_multimodal/data/filter_data/BMMC/RNA_filter.h5ad"
    # ),

    'HLCA_core': dict(
        dataset_name="HLCA_core",
        batch_key="donor_id",
        labels_key="cell_type",
        file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/HLCA_core_processed.h5ad",
    )

}


# training_params = {
#     'latent_dim16': dict(
#         latent_dim=16,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
#     'latent_dim32': dict(
#         latent_dim=32,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
#     'latent_dim64': dict(
#         latent_dim=64,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
#     'latent_dim128': dict(
#         latent_dim=128,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
#     'latent_dim256': dict(
#         latent_dim=256,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
#     'latent_dim512': dict(
#         latent_dim=512,
#         lr=1e-3,
#         batch_size=128,
#         beta__kl=0.0001,
#         epochs=150,
#         normaliztion='layernorm'
#     ),
# }


VAE_training_params = {
    'batch_size2048': dict(
        latent_dim=128,
        lr=1e-3,
        batch_size=128,
        beta__kl=0.01,
        epochs=150,
        normaliztion='batchnorm'
    ),
}

training_params = {
    'batch_size2048': dict(
        latent_dim=256,
        lr=1e-2,
        batch_size=2048,
        beta__kl=0.01,
        epochs=150,
        normaliztion='layernorm'
    ),
    'batch_size1024': dict(
        latent_dim=256,
        lr=1e-2,
        batch_size=1024,
        beta__kl=0.01,
        epochs=150,
        normaliztion='layernorm'
    ),
    'batch_size512': dict(
        latent_dim=256,
        lr=1e-3,
        batch_size=512,
        beta__kl=0.001,
        epochs=150,
        normaliztion='layernorm'
    ),
    'batch_size256': dict(
        latent_dim=256,
        lr=1e-3,
        batch_size=256,
        beta__kl=0.001,
        epochs=150,
        normaliztion='layernorm'
    ),
    'batch_size128': dict(
        latent_dim=256,
        lr=1e-3,
        batch_size=128,
        beta__kl=0.001,
        epochs=150,
        normaliztion='layernorm'
    ),
}
