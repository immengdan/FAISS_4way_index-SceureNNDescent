import numpy as np
from pynndescent import NNDescent
import os

class SecureNNDescentIndexer:
    def __init__(self, k=10, method="HE", decrypt=False, epsilon=0.5):
        self.k = k
        self.method = method
        self.decrypt = decrypt
        self.epsilon = epsilon
        self.original_xb = None

        if method == "SE":
            self.key = os.urandom(16)
        elif method == "OPE":
            self.ope_key = 0x1234ABCD

    def encrypt_vector(self, vec):
        if self.method == "HE":
            return vec.astype(np.float32)
        elif self.method == "SE":
            key_part = np.frombuffer(self.key[:4], dtype=np.float32)[0]
            return vec + key_part
        elif self.method == "Perturbation":
            noise = np.random.normal(0, 0.05 * np.ptp(vec), vec.shape)
            return vec + noise
        elif self.method == "OPE":
            return np.bitwise_xor((vec * 1000).astype(np.int32), self.ope_key).astype(np.float32)
        elif self.method == "DP":
            sensitivity = np.max(np.abs(np.diff(vec)))
            noise = np.random.laplace(0, sensitivity / self.epsilon, vec.shape)
            return vec + noise
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def decrypt_vector(self, vec):
        if not self.decrypt:
            raise RuntimeError("Decryption disabled in config")
        if self.method == "HE":
            return vec
        elif self.method == "SE":
            key_part = np.frombuffer(self.key[:4], dtype=np.float32)[0]
            return vec - key_part
        elif self.method == "OPE":
            return np.bitwise_xor(vec.astype(np.int32), self.ope_key).astype(np.float32) / 1000
        return vec

    def build(self, xb):
        self.original_xb = xb.copy()
        self.encrypted_xb = np.array([self.encrypt_vector(x) for x in xb])
        self.index = NNDescent(self.encrypted_xb, n_neighbors=self.k, metric="euclidean")

    def rerank_after_decryption(self, encrypted_query, topk_indices):
        query_vec = self.decrypt_vector(encrypted_query)
        reranked_indices = []
        for row in topk_indices:
            candidates = [self.decrypt_vector(self.encrypted_xb[i]) for i in row]
            dists = [np.linalg.norm(query_vec - cand) for cand in candidates]
            sorted_idx = np.argsort(dists)[:self.k]
            reranked_indices.append([row[i] for i in sorted_idx])
        return np.array(reranked_indices)

    def search(self, xq, k=None):
        if k is None:
            k = self.k
        encrypted_xq = np.array([self.encrypt_vector(x) for x in xq])
        indices, distances = self.index.query(encrypted_xq, k=k)

        if self.decrypt:
            reranked_all = []
            for query_enc, row in zip(encrypted_xq, indices):
                reranked_all.append(self.rerank_after_decryption(query_enc, [row])[0])
            return distances, np.array(reranked_all)

        return distances, indices
