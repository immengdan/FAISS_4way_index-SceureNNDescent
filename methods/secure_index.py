import numpy as np
from pynndescent import NNDescent
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import tenseal as ts 

class SecureNNDescent:
    def __init__(self, k=10, method="HE", decrypt=False, epsilon=0.5):
        self.k = k
        self.method = method
        self.decrypt = decrypt
        self.epsilon = epsilon  # For DP
        self.original_xb = None
        
        # Initialize crypto tools
        if method == "SE":
            self.key = os.urandom(32)  # AES-256 key
        elif method == "OPE":
            self.ope_key = os.urandom(32)  # For demo only, use KMS in production

    def encrypt_vector(self, vec):
        """标准化加密方法"""
        if self.method == "HE":
            context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
            return ts.bfv_vector(context, vec)
        
        elif self.method == "SE":
            nonce = os.urandom(16)
            cipher = Cipher(algorithms.AES(self.key), modes.CTR(nonce), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(vec.tobytes()) + encryptor.finalize()
            return np.frombuffer(nonce + ciphertext, dtype=np.uint8)
        
        elif self.method == "Perturbation":
            data_range = np.max(vec) - np.min(vec)
            noise = np.random.normal(0, 0.05 * data_range, vec.shape)
            return vec + noise
        
        elif self.method == "OPE":
            scaled = (vec * 1e6).astype(int)
            return np.bitwise_xor(scaled, 0xFFFF) 
        
        elif self.method == "DP":
            sensitivity = np.max(np.abs(np.diff(vec))) 
            noise = np.random.laplace(0, sensitivity/self.epsilon, vec.shape)
            return vec + noise
        
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def decrypt_vector(self, vec):
        """标准化解密方法"""
        if not self.decrypt:
            raise RuntimeError("Decryption disabled in config")
            
        if self.method == "HE":
            return vec.decrypt()
        
        elif self.method == "SE":
            nonce, ciphertext = vec[:16], vec[16:]
            cipher = Cipher(algorithms.AES(self.key), modes.CTR(nonce), backend=default_backend())
            decryptor = cipher.decryptor()
            return np.frombuffer(decryptor.update(ciphertext) + decryptor.finalize(), dtype=vec.dtype)
        
        elif self.method == "OPE":
            scaled = np.bitwise_xor(vec.astype(int), 0xFFFF)
            return scaled / 1e6
        
        # Perturbation和DP无法完全解密
        return vec  

    def build_encrypted(self, xb):
        self.original_xb = xb.copy()
        self.encrypted_xb = np.array([self.encrypt_vector(x) for x in xb])
        self.index = NNDescent(self.encrypted_xb, n_neighbors=self.k, metric="euclidean")

    def search_encrypted(self, xq):
        encrypted_xq = np.array([self.encrypt_vector(x) for x in xq])
        indices, distances = self.index.query(encrypted_xq, k=self.k)
        return indices if not self.decrypt else (
            indices, 
            distances,
            [self.decrypt_vector(self.encrypted_xb[i]) for i in indices]
        )