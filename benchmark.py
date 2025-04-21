import os
import time
import numpy as np
from utils.metrics import record_time, get_file_size_MB, measure_memory_peak
from utils.metrics import evaluate_results
from utils.recorder import BenchmarkRecorder
from methods.faiss_index import FAISSIndexer
from methods.secure_index import SecureNNDescent
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

    nq = 10000  # 假设查询集为 10K
    xq = load_fvecs_fbin(query_path, count=nq, dim=dim)

    offset = 0
    while offset < total_size:
        print(f"\n[Batch @ offset={offset}] Loading data...")
        size = min(batch_size, total_size - offset)
        xb = load_fvecs_fbin(base_path, offset=offset, count=size, dim=dim)
        
        # ===== 每个 batch 都重建 Ground Truth =====
        print("Building ground truth (Flat index)...")
        flat_index = FAISSIndexer(dim, "flat")
        flat_index.add_with_progress(xb)
        _, true_I = flat_index.search(xq, k)

        # ===== Phase 1: FAISS Baseline =====
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
            _, I = indexer.search(xq, k)
            query_time = time.time() - start
            recall = evaluate_results(true_I, I, k)
            
            index_path = f"saved_index/faiss_{index_type}.index"
            index_size_MB = get_file_size_MB(index_path)

            recorder.record_run(
                f"faiss_{index_type}",
                {"index_type": index_type},
                {
                    "query_time": query_time,
                    "recall": recall,
                    "index_time": index_time,
                    "index_memory_MB": index_memory_peak_MB,
                    "index_size_MB": index_size_MB
                },
                notes="FAISS baseline on original data"
            )

        # ===== Phase 2: NNDescent Plaintext =====
        print("\nRunning NNDescent baseline (plaintext)...")
        def build_plain_nnd():
            nndescent = NNDescentIndexer(n_neighbors=k)
            nndescent.build(xb)
            return nndescent
        
        nndescent, index_time = record_time(build_plain_nnd)
        index_memory_peak_MB = measure_memory_peak(build_plain_nnd)

        start = time.time()
        I_nndescent = nndescent.search(xq, k)
        query_time = time.time() - start
        recall = evaluate_results(true_I, I_nndescent, k)

        index_path = "saved_index/nndescent_plain.pkl"
        index_size_MB = get_file_size_MB(index_path)

        recorder.record_run(
            "nndescent_plain",
            {"method": "nndescent", "encrypted": False},
            {
                "query_time": query_time,
                "recall": recall,
                "index_time": index_time,
                "index_memory_MB": index_memory_peak_MB,
                "index_size_MB": index_size_MB
           },
            notes="NNDescent baseline on original data"
        )

        # ===== Phase 3: Secure NNDescent + 可选解密 =====
        encryption_methods = ["HE", "SE", "Perturbation", "OPE", "DP"]

        for method in encryption_methods:
            for decrypt in [False, True]:
                print(f"\nRunning Secure NNDescent with encryption: {method}, decrypt: {decrypt}")
                def build_secure_nnd():
                    secure_nnd = SecureNNDescent(k=k, method=method, decrypt=decrypt)
                    secure_nnd.build_encrypted(xb)
                    return secure_nnd
                
                secure_nnd, index_time = record_time(build_secure_nnd)
                index_memory_peak_MB = measure_memory_peak(build_secure_nnd)
                
                encryption_time = secure_nnd.encryption_tiem if hasattr(secure_nnd, "encryption_time") else 0
                decryption_time = secure_nnd.decryption_time if (decrypt and hasattr(secure_nnd, "decryption_time")) else 0
                
                start = time.time()
                encrypted_I = secure_nnd.search_encrypted(xq)
                query_time = time.time() - start
                
                recall = evaluate_results(true_I, encrypted_I, k)
                
                index_path = f"saved_index/nndescent_{method}.pkl"
                index_size_MB = get_file_size_MB(index_path)

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
                        "encryption_time": encryption_time,
                        "index_time": index_time,
                        "decryption_time": decryption_time,
                        "index_memory_MB": index_memory_peak_MB,
                        "index_size_MB": index_size_MB
                    },
                    notes=f"Secure NNDescent using {method}, decrypt={decrypt}"
                )

        # 更新 offset
        offset += size

    recorder.save_all()
    print("\n=== Benchmark Finished ===")

if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "data/base.10M.fbin")
    query_path = os.path.join(os.path.dirname(__file__), "data/query.public.10K.fbin")

    total_size = 10_000_000  # 总数据量
    batch_size = 1_000_000  # 每批次处理的数据量
    dim = 96  # 向量维度

    run_benchmark_in_batches(base_path, query_path, total_size, batch_size, k=10, dim=dim)
