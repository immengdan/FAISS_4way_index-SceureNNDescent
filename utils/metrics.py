import os
import time
import psutil
import tracemalloc
from memory_profiler import memory_usage
import numpy as np  

def evaluate_results(true_indices, retrieved_indices, k):
    correct = 0
    for true, retrieved in zip(true_indices, retrieved_indices):
        correct += len(set(true) & set(retrieved))
    total = len(true_indices) * k
    return correct / total

def evaluate_precision(true_indices, retrieved_indices):
    correct = 0
    total = 0
    for true, retrieved in zip(true_indices, retrieved_indices):
        correct += len(set(true) & set(retrieved))
        total += len(retrieved)
    return correct / total if total > 0 else 0

def mean_topk_distance(distances):
    return np.mean(distances)

def average_query_time(total_time, num_queries):
    return total_time / num_queries

def measure_encryption_time(encrypt_func, xb):
    start = time.time()
    [encrypt_func(x) for x in xb]
    return time.time() - start

def record_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed

def get_file_size_MB(file_path):
    if not os.path.exists(file_path):
        return 0.0
    size_bytes = os.path.getsize(file_path)
    return round(size_bytes / (1024 * 1024), 2)

def measure_memory_peak(func, *args, **kwargs):
    mem_usage = memory_usage((func, args, kwargs), max_usage=True, interval=0.1, timeout=None)
    return mem_usage
