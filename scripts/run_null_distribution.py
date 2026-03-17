"""Compute null distribution from random gene knockouts across multiple GPUs.

Perturbs N random non-ALS genes at knockout dose on a subset of disease cells,
embeds each, and computes cosine distance from baseline. Saves the null
distribution for statistical testing in Task 3.

Usage:
    python scripts/run_null_distribution.py \
        --data_path data/counts_combined_filtered_BA4_sALS_PN.h5ad \
        --baseline_emb data/embeddings_baseline.npy \
        --output_path data/null_distribution.npy \
        --n_random 50 \
        --max_cells 2000 \
        --n_gpus 8
"""

import argparse
import numpy as np
import os
import time
from scipy import sparse
import torch.multiprocessing as mp

ALS_GENES = [
    'TARDBP', 'SOD1', 'FUS', 'C9orf72',
    'STMN2', 'ATXN2', 'UNC13A',
    'POU3F1', 'SCN4B',
    'OPTN', 'TBK1', 'NEK1',
]


def perturb_gene(adata, gene_name, factor):
    adata_pert = adata.copy()
    gene_idx = list(adata_pert.var_names).index(gene_name)
    if sparse.issparse(adata_pert.X):
        X = adata_pert.X.tocsc()
        col = X.getcol(gene_idx).toarray().flatten() * factor
        col = np.round(col).astype(X.dtype)
        X_lil = X.tolil()
        X_lil[:, gene_idx] = col.reshape(-1, 1)
        adata_pert.X = X_lil.tocsr()
    else:
        adata_pert.X[:, gene_idx] = np.round(adata_pert.X[:, gene_idx] * factor).astype(adata.X.dtype)
    return adata_pert


def worker(gpu_id, gene_list, data_path, emb_baseline, disease_mask, max_cells, output_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import scanpy as sc
    from helical.models.geneformer import Geneformer, GeneformerConfig

    adata = sc.read_h5ad(data_path)
    adata_disease = adata[disease_mask].copy()
    emb_disease = emb_baseline[disease_mask]

    rng = np.random.RandomState(42)
    if adata_disease.shape[0] > max_cells:
        idx = rng.choice(adata_disease.shape[0], max_cells, replace=False)
        adata_sub = adata_disease[idx].copy()
        emb_sub = emb_disease[idx]
    else:
        adata_sub = adata_disease
        emb_sub = emb_disease

    gf = Geneformer(GeneformerConfig(model_name='gf-12L-95M-i4096', device='cuda:0', batch_size=16))

    shifts = []
    for i, gene in enumerate(gene_list):
        try:
            adata_pert = perturb_gene(adata_sub, gene, 0.0)
            dataset = gf.process_data(adata_pert)
            emb_pert = gf.get_embeddings(dataset)

            norm_b = emb_sub / (np.linalg.norm(emb_sub, axis=1, keepdims=True) + 1e-8)
            norm_p = emb_pert / (np.linalg.norm(emb_pert, axis=1, keepdims=True) + 1e-8)
            cos_dist = 1 - np.sum(norm_b * norm_p, axis=1)

            euc_shift = np.linalg.norm(emb_pert - emb_sub, axis=1)

            shifts.append({
                'gene': gene,
                'mean_cosine_dist': cos_dist.mean(),
                'mean_euc_shift': euc_shift.mean(),
            })
            print(f"[GPU {gpu_id}] ({i+1}/{len(gene_list)}) {gene}: cos={cos_dist.mean():.6f} euc={euc_shift.mean():.4f}")
        except Exception as e:
            print(f"[GPU {gpu_id}] Skipping {gene}: {e}")

    np.save(os.path.join(output_dir, f'null_shard_{gpu_id}.npy'), shifts)
    print(f"[GPU {gpu_id}] Done: {len(shifts)} genes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--baseline_emb', required=True)
    parser.add_argument('--output_path', default='data/null_distribution.npy')
    parser.add_argument('--n_random', type=int, default=50)
    parser.add_argument('--max_cells', type=int, default=2000)
    parser.add_argument('--n_gpus', type=int, default=8)
    args = parser.parse_args()

    import scanpy as sc

    print("Preparing null distribution jobs...")
    adata = sc.read_h5ad(args.data_path)
    emb_baseline = np.load(args.baseline_emb)

    disease_mask = (adata.obs['Condition'] == 'ALS').values

    als_set = set(ALS_GENES)
    all_genes = list(adata.var_names)
    non_als = [g for g in all_genes if g not in als_set]

    if sparse.issparse(adata.X):
        pct_nz = np.array((adata.X > 0).mean(axis=0)).flatten()
    else:
        pct_nz = np.mean(adata.X > 0, axis=0)
    expressed = [g for g in non_als if pct_nz[all_genes.index(g)] > 0.05]

    rng = np.random.RandomState(123)
    random_genes = list(rng.choice(expressed, min(args.n_random, len(expressed)), replace=False))
    print(f"Selected {len(random_genes)} random expressed genes for null distribution")

    n_gpus = min(args.n_gpus, len(random_genes))
    gpu_genes = [[] for _ in range(n_gpus)]
    for i, gene in enumerate(random_genes):
        gpu_genes[i % n_gpus].append(gene)

    mp.set_start_method('spawn', force=True)
    t0 = time.time()
    procs = []
    for gpu_id in range(n_gpus):
        p = mp.Process(target=worker, args=(
            gpu_id, gpu_genes[gpu_id], args.data_path, emb_baseline,
            disease_mask, args.max_cells, os.path.dirname(args.output_path) or 'data'
        ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    elapsed = time.time() - t0
    print(f"\nAll workers done in {elapsed/60:.1f} minutes")

    # Merge
    all_shifts = []
    output_dir = os.path.dirname(args.output_path) or 'data'
    for gpu_id in range(n_gpus):
        path = os.path.join(output_dir, f'null_shard_{gpu_id}.npy')
        if os.path.exists(path):
            shard = np.load(path, allow_pickle=True)
            all_shifts.extend(shard)
            os.remove(path)

    np.save(args.output_path, all_shifts)
    print(f"Saved null distribution ({len(all_shifts)} genes) to {args.output_path}")

    cos_dists = [s['mean_cosine_dist'] for s in all_shifts]
    euc_shifts = [s['mean_euc_shift'] for s in all_shifts]
    print(f"Cosine dist: mean={np.mean(cos_dists):.6f}, std={np.std(cos_dists):.6f}")
    print(f"Euclidean shift: mean={np.mean(euc_shifts):.4f}, std={np.std(euc_shifts):.4f}")


if __name__ == '__main__':
    main()
