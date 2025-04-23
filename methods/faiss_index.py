import faiss
import numpy as np
from tqdm import tqdm

class FAISSIndexer:
    def __init__(self, dim, index_type="flat"):
        self.dim = dim
        self.index_type = index_type.lower()

        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(dim)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 100)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, 32)
        elif self.index_type == "lsh":
            self.index = faiss.IndexLSH(dim, 128)
        else:
            raise ValueError("Unsupported index type")

    def train(self, xb):
        if hasattr(self.index, "train"):
            self.index.train(xb)

    def add_with_progress(self, xb):
        for i in tqdm(range(0, len(xb), 100000), desc="Adding to index"):
            self.index.add(xb[i:i+100000])

    def search(self, xq, k):
        distances, indices = self.index.search(xq, k)
        return indices, distances
