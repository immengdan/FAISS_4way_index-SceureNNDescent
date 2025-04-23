import os
import time
import numpy as np
from utils.metrics import record_time, get_file_size_MB, measure_memory_peak
from utils.metrics import evaluate_results, evaluate_precision, mean_topk_distance
from utils.recorder import BenchmarkRecorder
from methods.faiss_index import FAISSIndexer
from methods.secure_index import SecureNNDescentIndexer
from methods.nndescent_index import NNDescentIndexer 

def load_fvecs_fbin(filename, offset=0, count=None, dim=96):
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
    with open(filename, 'rb') as f:
        f.seek(offset * dim * itemsize)
        if count is not None:
            data = np.fromfile(f, dtype=dtype, count=count * dim)
            data = data.reshape(-1, dim)
        else:
            data = np.fromfile(f, dtype=dtype)
            data = data.reshape(-1, dim)
    return data

def run_benchmark_in_batches(base_path, query_path, total_size, batch_size, k=10, dim=96):
    recorder = BenchmarkRecorder("faiss_nndescent_benchmark")

    nq = 10000
    xq = load_fvecs_fbin(query_path, count=nq, dim=dim)

    offset = 0
    while offset < total_size:
        print(f"\n[Batch @ offset={offset}] Loading data...")
        size = min(batch_size, total_size - offset)
        xb = load_fvecs_fbin(base_path, offset=offset, count=size, dim=dim)
        
        # ===== Ground Truth =====
        print("Building ground truth (Flat index)...")
        flat_index = FAISSIndexer(dim, "flat")
        flat_index.add_with_progress(xb)
        _, true_I = flat_index.search(xq, k)

        # ===== FAISS Baselines =====
        for index_type in ["flat", "ivf", "hnsw", "lsh"]:
            print(f"\nRunning FAISS baseline: {index_type}")
            indexer = FAISSIndexer(dim, index_type)
            def build_faiss():
                if index_type == "ivf":
                    indexer.train(xb)
                indexer.add_with_progress(xb)
                return indexer

            indexer, index_time = record_time(build_faiss)
            index_memory_peak_MB = measure_memory_peak(build_faiss)

            start = time.time()
            D, I = indexer.search(xq, k)
            query_time = time.time() - start

            recall = evaluate_results(true_I, I, k)
            precision = evaluate_precision(true_I, I)
            mean_distance = mean_topk_distance(D)

            index_path = f"saved_index/faiss_{index_type}.index"
            index_size_MB = get_file_size_MB(index_path)

            recorder.record_run(
                f"faiss_{index_type}",
                {"index_type": index_type},
                {
                    "query_time": query_time,
                    "recall": recall,
                    "precision": precision,
                    "mean_distance": mean_distance,
                    "index_time": index_time,
                    "index_memory_MB": index_memory_peak_MB,
                    "index_size_MB": index_size_MB
                },
                notes="FAISS baseline on original data"
            )

        # ===== NNDescent Baseline (Plain) =====
        print("\nRunning NNDescent baseline (plaintext)...")
        def build_plain_nnd():
            nndescent = NNDescentIndexer(n_neighbors=k)
            nndescent.build(xb)
            return nndescent

        nndescent, index_time = record_time(build_plain_nnd)
        index_memory_peak_MB = measure_memory_peak(build_plain_nnd)

        start = time.time()
        D_nndescent, I_nndescent = nndescent.search(xq, k)
        query_time = time.time() - start

        recall = evaluate_results(true_I, I_nndescent, k)
        precision = evaluate_precision(true_I, I_nndescent)
        mean_distance = mean_topk_distance(D_nndescent)

        index_path = "saved_index/nndescent_plain.pkl"
        index_size_MB = get_file_size_MB(index_path)

        recorder.record_run(
            "nndescent_plain",
            {"method": "nndescent", "encrypted": False},
            {
                "query_time": query_time,
                "recall": recall,
                "precision": precision,
                "mean_distance": mean_distance,
                "index_time": index_time,
                "index_memory_MB": index_memory_peak_MB,
                "index_size_MB": index_size_MB
           },
            notes="NNDescent baseline on original data"
        )

        # ===== Secure NNDescent Variants =====
        encryption_methods = ["HE", "SE", "Perturbation", "OPE", "DP"]
        for method in encryption_methods:
            for decrypt in [False, True]:
                print(f"\nRunning Secure NNDescent with encryption: {method}, decrypt: {decrypt}")

                def build_secure_nnd():
                    secure_nnd = SecureNNDescentIndexer(k=k, method=method, decrypt=decrypt)
                    secure_nnd.build(xb)
                    return secure_nnd

                secure_nnd, index_time = record_time(build_secure_nnd)
                index_memory_peak_MB = measure_memory_peak(build_secure_nnd)

                encryption_time = 0   
                decryption_time = 0  
                start = time.time()
                D_encrypted, I_encrypted = secure_nnd.search(xq, k=k)
                query_time = time.time() - start

                recall = evaluate_results(true_I, I_encrypted, k)
                precision = evaluate_precision(true_I, I_encrypted)
                mean_distance = mean_topk_distance(D_encrypted)
                index_size_MB = 0

                recorder.record_run(
                    f"nndescent_secure_{method.lower()}_{'dec' if decrypt else 'nodec'}",
                    {
                        "method": "nndescent",
                        "encrypted": True,
                        "encryption": method,
                        "decryption": decrypt
                    },
                    {
                        "query_time": query_time,
                        "recall": recall,
                        "precision": precision,
                        "mean_distance": mean_distance,
                        "encryption_time": encryption_time,
                        "index_time": index_time,
                        "decryption_time": decryption_time,
                        "index_memory_MB": index_memory_peak_MB,
                        "index_size_MB": index_size_MB
                    },
                    notes=f"Secure NNDescent using {method}, decrypt={decrypt}"
                )

        offset += size
        recorder.save_all()
        print("\n=== Benchmark Finished ===")

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "data/base.10M.fbin")
    query_path = os.path.join(os.path.dirname(__file__), "data/query.public.10K.fbin")
    total_size = 10_000_000
    batch_size = 1_000_000
    dim = 96
    run_benchmark_in_batches(base_path, query_path, total_size, batch_size, k=10, dim=dim)
