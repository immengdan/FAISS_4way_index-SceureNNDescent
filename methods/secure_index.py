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
            self.key = os.urandom(4096) 
        elif method == "OPE":
            self.ope_scale = 100.0
            self.ope_offset = 1000.0
            
        self.global_sensitivity = 1.0
        
        self.perturbation_scale = 0.01

    def _safe_encrypt(self, vec):

        vec = np.asarray(vec, dtype=np.float32)
        try:
            encrypted = self._raw_encrypt(vec)
            encrypted = np.nan_to_num(
                encrypted,
                nan=0.0,
                posinf=np.finfo(np.float32).max,
                neginf=np.finfo(np.float32).min
            )
            return encrypted
        except Exception as e:
            print(f"Fail: {e}")
            return vec  

    def _raw_encrypt(self, vec):
        if self.method == "HE":
            return vec + np.random.normal(0, 1e-4, vec.shape) 
            
        elif self.method == "SE":
            key_len = len(vec) * 4  
            if len(self.key) < key_len:
                self.key = os.urandom(key_len) 
            key_parts = np.frombuffer(self.key[:key_len], dtype=np.float32)
            return vec + key_parts
            
        elif self.method == "Perturbation":
            noise = np.random.normal(0, self.perturbation_scale, vec.shape)
            return vec + noise
            
        elif self.method == "OPE":
            scaled = (vec * self.ope_scale + self.ope_offset)
            return np.clip(scaled, -1e6, 1e6)
            
        elif self.method == "DP":
            
            scale = self.global_sensitivity / self.epsilon
            noise = np.random.laplace(loc=0.0, scale=scale, size=vec.shape)
            return vec + noise
        else:
            raise ValueError(f"Unknown: {self.method}")

    def encrypt_vector(self, vec):
        return self._safe_encrypt(vec)

    def decrypt_vector(self, vec):
        if not self.decrypt:
            raise RuntimeError("Forbidden")
            
        vec = np.asarray(vec, dtype=np.float32)
        try:
            if self.method == "SE":
                key_len = len(vec) * 4
                key_parts = np.frombuffer(self.key[:key_len], dtype=np.float32)
                return vec - key_parts
                
            elif self.method == "OPE":
                return (vec - self.ope_offset) / self.ope_scale
                
            return vec 
            
        except Exception as e:
            print(f"Failed: {e}")
            return vec

    def build(self, xb):
        self.original_xb = xb.copy()
        self.encrypted_xb = np.array([self.encrypt_vector(x) for x in xb])
        
        assert not np.any(np.isnan(self.encrypted_xb)), "Include NaN!"
        assert not np.any(np.isinf(self.encrypted_xb)), "Include Inf!"
        
        self.index = NNDescent(
            self.encrypted_xb, 
            n_neighbors=self.k, 
            metric="euclidean",
            random_state=42
        )

    def search(self, xq, k=None):
        if k is None:
            k = self.k
            
        encrypted_xq = np.array([self.encrypt_vector(x) for x in xq])
        indices, distances = self.index.query(encrypted_xq, k=k)

        if self.decrypt:
            decrypted_results = []
            for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                original_query = xq[i]
            
                decrypted_candidates = [self.decrypt_vector(self.encrypted_xb[idx]) 
                                        for idx in idx_row]
            
                true_distances = [np.linalg.norm(original_query - cand) 
                                  for cand in decrypted_candidates]
                sorted_order = np.argsort(true_distances)[:k]
            
                decrypted_results.append((
                    np.array(true_distances)[sorted_order], 
                    idx_row[sorted_order]
                ))

            final_distances = np.array([d[0] for d in decrypted_results])
            final_indices = np.array([d[1] for d in decrypted_results])
            return final_distances, final_indices
        
        return distances, indices
    