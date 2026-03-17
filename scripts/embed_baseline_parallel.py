"""Embed all cells in parallel across multiple GPUs."""
import numpy as np
import os
import torch.multiprocessing as mp


def embed_shard(gpu_id, n_gpus, data_path, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import scanpy as sc
    from helical.models.geneformer import Geneformer, GeneformerConfig

    adata = sc.read_h5ad(data_path)
    n = adata.shape[0]
    start = (n * gpu_id) // n_gpus
    end = (n * (gpu_id + 1)) // n_gpus
    adata_shard = adata[start:end].copy()

    gf = Geneformer(GeneformerConfig(model_name='gf-12L-95M-i4096', device='cuda:0', batch_size=16))
    dataset = gf.process_data(adata_shard)
    emb = gf.get_embeddings(dataset)
    np.save(f'{output_dir}/emb_shard_{gpu_id}.npy', emb)
    print(f'[GPU {gpu_id}] Done: cells {start}-{end}, shape {emb.shape}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    N_GPUS = 8
    DATA_PATH = 'data/counts_combined_filtered_BA4_sALS_PN.h5ad'
    OUTPUT_DIR = 'data'

    procs = []
    for i in range(N_GPUS):
        p = mp.Process(target=embed_shard, args=(i, N_GPUS, DATA_PATH, OUTPUT_DIR))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    shards = [np.load(f'{OUTPUT_DIR}/emb_shard_{i}.npy') for i in range(N_GPUS)]
    emb = np.concatenate(shards, axis=0)
    np.save(f'{OUTPUT_DIR}/embeddings_baseline.npy', emb)
    for i in range(N_GPUS):
        os.remove(f'{OUTPUT_DIR}/emb_shard_{i}.npy')
    print(f'Merged: {emb.shape}')
