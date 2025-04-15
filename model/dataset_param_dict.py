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
    #     file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/BMMC_RNA_filter.h5ad"
    # ),
    # 'PBMC_': dict(
    #     # dataset
    #     dataset_name="PBMC",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/PBMC_RNA_filter.h5ad"
    # ),


    'immune*20': dict(
        dataset_name="immune*20",
        batch_key="batch",
        labels_key="final_annotation",
        file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/immune_processed*20.h5ad"
    ),

    'BMMC*20': dict(
        # dataset
        dataset_name="BMMC*20",
        batch_key="batch",
        labels_key="cell_type",
        file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/BMMC_RNA_filter*20.h5ad"
    ),

    'fetal_lung*20': dict(
        # dataset
        dataset_name="fetal_lung*20",
        batch_key="batch",
        labels_key="broad_celltype",
        file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/fetal_lung_processed*20.h5ad"
    ),

    'HLCA_core*20': dict(
        dataset_name="HLCA_core*20",
        batch_key="donor_id",
        labels_key="cell_type",
        file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/HLCA_core_processed*20.h5ad",
    ),
    'pancreas*20': dict(
        dataset_name="pancreas*20",
        batch_key="tech",
        labels_key="celltype",
        file_path="/mnt/zhangzheng_group/xuany-54/QVAE/data/pancreas_processed*20.h5ad"
    ),

}
