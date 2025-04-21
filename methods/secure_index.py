import numpy as np
from pynndescent import NNDescent

class SecureNNDescent:
    def __init__(self, k=10, method="HE", decrypt=False):
        self.k = k
        self.method = method
        self.decrypt = decrypt

    def encrypt_vector(self, vec):
        if self.method == "HE":
            return vec + 1e-3
        elif self.method == "SE":
            return vec * 1.01
        elif self.method == "Perturbation":
            return vec + np.random.normal(0, 0.1, size=vec.shape)
        elif self.method == "OPE":
            return np.sort(vec)
        elif self.method == "DP":
            return vec + np.random.laplace(0, 0.05, size=vec.shape)
        else:
            return vec

    def decrypt_vector(self, vec):
        if self.method == "HE":
            return vec - 1e-3
        elif self.method == "SE":
            return vec / 1.01
        else:
            return vec

    def build_encrypted(self, xb):
        self.encrypted_xb = np.array([self.encrypt_vector(x) for x in xb])
        self.index = NNDescent(self.encrypted_xb, n_neighbors=self.k, metric="euclidean")

    def search_encrypted(self, xq):
        if self.decrypt:
            xq = np.array([self.encrypt_vector(x) for x in xq])
            xq = np.array([self.decrypt_vector(x) for x in xq])
        else:
            xq = np.array([self.encrypt_vector(x) for x in xq])
        indices, _ = self.index.query(xq, k=self.k)
        return indices
