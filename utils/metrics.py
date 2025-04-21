import os
import time
import psutil
import tracemalloc
from memory_profiler import memory_usage

def evaluate_results(true_indices, retrieved_indices, k):
    """
    Evaluate recall@k between true indices and retrieved indices.
    """
    correct = 0
    for true, retrieved in zip(true_indices, retrieved_indices):
        correct += len(set(true) & set(retrieved))
    total = len(true_indices) * k
    return correct / total

def record_time(func, *args, **kwargs):
    """记录函数运行时间"""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed

def get_file_size_MB(file_path):
    """获取文件大小（单位：MB）"""
    if not os.path.exists(file_path):
        return 0.0
    size_bytes = os.path.getsize(file_path)
    return round(size_bytes / (1024 * 1024), 2)

def measure_memory_peak(func, *args, **kwargs):
    """记录函数运行的最大内存使用（MB）"""
    mem_usage = memory_usage((func, args, kwargs), max_usage=True, interval=0.1, timeout=None)
    return mem_usage  # 返回 MB
