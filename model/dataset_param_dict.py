
dataset_params = {
    'pancreas': dict(
    # dataset
    dataset_name="pancreas",
    batch_key="tech",
    labels_key="celltype",
    file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/pancreas_processed.h5ad"),

    # 'BMMC_': dict(
    #     # dataset
    #     dataset_name="BMMC",
    #     batch_key="batch",
    #     labels_key="cell_type",
    #     file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/BMMC_processed.h5ad"
    # ),
    'fetal_lung': dict(
        # dataset
        dataset_name="fetal_lung",
        batch_key="batch",
        labels_key="broad_celltype",
        file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/fetal_lung_processed.h5ad"
    ),

    'immune': dict(
        dataset_name="immune",
        batch_key="batch",
        labels_key="final_annotation",
        file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/immune_processed.h5ad"
    ),
    'BMMC_multiome': dict(
        dataset_name="BMMC_multiome",
        batch_key="batch",
        labels_key="cell_type",
        file_path="/data2/wfa/project/single_cell_multimodal/data/filter_data/BMMC/RNA_filter.h5ad"
    ),

    'HLCA_core': dict(
        dataset_name="HLCA_core",
        batch_key="donor_id",
        labels_key="cell_type",
        file_path="/data2/wfa/scMulti-omics/QVAE/scRNA/HLCA_core_processed.h5ad",
    )

}
