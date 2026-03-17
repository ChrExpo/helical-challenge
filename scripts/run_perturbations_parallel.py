"""Parallel perturbation runner across multiple GPUs.

Distributes gene×dose combinations across available GPUs.
Each GPU loads its own GeneFormer model and processes its share.

Usage:
    python scripts/run_perturbations_parallel.py \
        --data_path data/counts_combined_filtered_BA4_sALS_PN.h5ad \
        --baseline_emb data/embeddings_baseline.npy \
        --output_dir data \
        --n_gpus 8 \
        --max_cells 20000 \
        --condition disease  # or 'healthy' or 'both'
"""

import argparse
import numpy as np
import pickle
import os
import time
from scipy import sparse
import torch
import torch.multiprocessing as mp


DOSE_LEVELS = {
    'knockout':      0.0,
    'strong_down':   0.1,
    'moderate_down': 0.5,
    'mild_up':       1.5,
    'moderate_up':   2.0,
    'strong_up':     5.0,
    'extreme_up':   10.0,
}

ALS_GENES_CANDIDATES = [
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


def compute_effect(emb_baseline, emb_perturbed):
    norm_b = emb_baseline / (np.linalg.norm(emb_baseline, axis=1, keepdims=True) + 1e-8)
    norm_p = emb_perturbed / (np.linalg.norm(emb_perturbed, axis=1, keepdims=True) + 1e-8)
    cosine_dist = 1 - np.sum(norm_b * norm_p, axis=1)
    shift_vectors = emb_perturbed - emb_baseline
    shift_magnitude = np.linalg.norm(shift_vectors, axis=1)
    return {
        'cosine_distance': cosine_dist,
        'shift_magnitude': shift_magnitude,
        'mean_shift_vector': shift_vectors.mean(axis=0),  # 512-dim, for reversal scoring
    }


def worker(gpu_id, jobs, adata_path, emb_baseline, output_dir):
    """Worker function: loads GeneFormer on assigned GPU and processes jobs."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda:0'

    from helical.models.geneformer import Geneformer, GeneformerConfig
    import scanpy as sc

    print(f"[GPU {gpu_id}] Loading GeneFormer...")
    gf = Geneformer(GeneformerConfig(model_name="gf-12L-95M-i4096", device=device, batch_size=64))
    print(f"[GPU {gpu_id}] Loading data...")
    adata = sc.read_h5ad(adata_path)

    results = {}
    for i, (gene, dose_name, factor, cell_mask_path) in enumerate(jobs):
        t0 = time.time()

        cell_mask = np.load(cell_mask_path)
        adata_sub = adata[cell_mask].copy()
        emb_sub = emb_baseline[cell_mask]

        adata_pert = perturb_gene(adata_sub, gene, factor)
        dataset = gf.process_data(adata_pert)
        emb_pert = gf.get_embeddings(dataset)

        effect = compute_effect(emb_sub, emb_pert)
        effect['factor'] = factor

        key = f"{gene}__{dose_name}"
        results[key] = effect

        elapsed = time.time() - t0
        print(f"[GPU {gpu_id}] ({i+1}/{len(jobs)}) {gene} | {dose_name} (x{factor}) "
              f"cos_dist={effect['cosine_distance'].mean():.4f} [{elapsed:.1f}s]")

    out_path = os.path.join(output_dir, f"results_gpu{gpu_id}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"[GPU {gpu_id}] Done. Saved {len(results)} results to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--baseline_emb', required=True)
    parser.add_argument('--output_dir', default='data')
    parser.add_argument('--n_gpus', type=int, default=8)
    parser.add_argument('--max_cells', type=int, default=20000)
    parser.add_argument('--condition', choices=['disease', 'healthy', 'both'], default='both')
    args = parser.parse_args()

    import scanpy as sc

    print("Loading data for job preparation...")
    adata = sc.read_h5ad(args.data_path)
    emb_baseline = np.load(args.baseline_emb)

    available_genes = [g for g in ALS_GENES_CANDIDATES if g in adata.var_names]
    print(f"Available ALS genes: {available_genes}")

    disease_col = 'Condition'
    healthy_label = 'PN'
    disease_label = 'ALS'

    healthy_mask = (adata.obs[disease_col] == healthy_label).values
    disease_mask = (adata.obs[disease_col] == disease_label).values
    print(f"Healthy: {healthy_mask.sum():,}, Disease: {disease_mask.sum():,}")

    # Subsample masks
    rng = np.random.RandomState(42)
    conditions = []
    if args.condition in ('disease', 'both'):
        dm = disease_mask.copy()
        if dm.sum() > args.max_cells:
            idx = np.where(dm)[0]
            keep = rng.choice(idx, args.max_cells, replace=False)
            dm = np.zeros(len(dm), dtype=bool)
            dm[keep] = True
        mask_path = os.path.join(args.output_dir, 'mask_disease.npy')
        np.save(mask_path, dm)
        conditions.append(('disease', mask_path))

    if args.condition in ('healthy', 'both'):
        hm = healthy_mask.copy()
        max_h = min(args.max_cells, 10000)
        if hm.sum() > max_h:
            idx = np.where(hm)[0]
            keep = rng.choice(idx, max_h, replace=False)
            hm = np.zeros(len(hm), dtype=bool)
            hm[keep] = True
        mask_path = os.path.join(args.output_dir, 'mask_healthy.npy')
        np.save(mask_path, hm)
        conditions.append(('healthy', mask_path))

    # Build job list: (gene, dose_name, factor, mask_path)
    all_jobs = []
    for cond_name, mask_path in conditions:
        for gene in available_genes:
            for dose_name, factor in DOSE_LEVELS.items():
                all_jobs.append((gene, f"{cond_name}__{dose_name}", factor, mask_path))

    print(f"\nTotal jobs: {len(all_jobs)} ({len(available_genes)} genes × {len(DOSE_LEVELS)} doses × {len(conditions)} conditions)")

    # Distribute across GPUs
    n_gpus = min(args.n_gpus, len(all_jobs))
    gpu_jobs = [[] for _ in range(n_gpus)]
    for i, job in enumerate(all_jobs):
        gpu_jobs[i % n_gpus].append(job)

    for i in range(n_gpus):
        print(f"  GPU {i}: {len(gpu_jobs[i])} jobs")

    # Launch workers
    mp.set_start_method('spawn', force=True)
    processes = []
    t_start = time.time()

    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=worker,
            args=(gpu_id, gpu_jobs[gpu_id], args.data_path, emb_baseline, args.output_dir),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - t_start
    print(f"\nAll workers done in {elapsed/60:.1f} minutes")

    # Merge results
    print("Merging results...")
    results_disease = {}
    results_healthy = {}

    for gpu_id in range(n_gpus):
        path = os.path.join(args.output_dir, f"results_gpu{gpu_id}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            gpu_results = pickle.load(f)

        for key, effect in gpu_results.items():
            gene, cond_dose = key.split('__', 1)
            cond, dose_name = cond_dose.split('__', 1)

            if cond == 'disease':
                if gene not in results_disease:
                    results_disease[gene] = {}
                results_disease[gene][dose_name] = effect
            elif cond == 'healthy':
                if gene not in results_healthy:
                    results_healthy[gene] = {}
                results_healthy[gene][dose_name] = effect

        os.remove(path)

    # Save merged results
    with open(os.path.join(args.output_dir, 'results_disease.pkl'), 'wb') as f:
        pickle.dump(results_disease, f)
    with open(os.path.join(args.output_dir, 'results_healthy.pkl'), 'wb') as f:
        pickle.dump(results_healthy, f)

    print(f"Saved: results_disease.pkl ({len(results_disease)} genes), results_healthy.pkl ({len(results_healthy)} genes)")

    # Compute reversal scores
    centroid_healthy = emb_baseline[healthy_mask].mean(axis=0)
    centroid_disease = emb_baseline[disease_mask].mean(axis=0)
    d2h = centroid_healthy - centroid_disease
    d2h_norm = d2h / (np.linalg.norm(d2h) + 1e-8)

    np.save(os.path.join(args.output_dir, 'centroid_healthy.npy'), centroid_healthy)
    np.save(os.path.join(args.output_dir, 'centroid_disease.npy'), centroid_disease)
    np.save(os.path.join(args.output_dir, 'disease_to_healthy_direction.npy'), d2h_norm)

    reversal_rows = []
    for gene in results_disease:
        for dose_name, effect in results_disease[gene].items():
            proj = effect['shift_vectors'] @ d2h_norm
            reversal_rows.append({
                'gene': gene, 'dose': dose_name, 'factor': effect['factor'],
                'mean_reversal': proj.mean(), 'std_reversal': proj.std(),
                'pct_toward_healthy': (proj > 0).mean() * 100,
                'mean_cosine_dist': effect['cosine_distance'].mean(),
            })

    import pandas as pd
    df = pd.DataFrame(reversal_rows)
    df.to_csv(os.path.join(args.output_dir, 'reversal_scores.csv'), index=False)
    print(f"Saved reversal_scores.csv ({len(df)} rows)")
    print("\nDone! Results ready for Task 3 and Task 4 notebooks.")


if __name__ == '__main__':
    main()
