import os
from pynndescent import NNDescent
import numpy as np
import time

class NNDescentIndexer:
    def __init__(self, n_neighbors=10, n_jobs=4, verbose=False):
        """
        Initialize NNDescent indexer with configurable parameters
        
        Parameters:
        - n_neighbors: Number of neighbors for graph construction
        - n_jobs: Number of parallel jobs
        - verbose: Whether to print progress messages
        """
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.verbose = verbose
        # Set thread control environment variables
        os.environ["OMP_NUM_THREADS"] = str(n_jobs)
        os.environ["MKL_NUM_THREADS"] = str(n_jobs)
        self.index = None
        self.data = None

    def build(self, xb):
        """
        Build the NNDescent index
        
        Parameters:
        - xb: Base data vectors (numpy array)
        """
        if not isinstance(xb, np.ndarray):
            raise ValueError("Input data must be a numpy array")
            
        if len(xb.shape) != 2:
            raise ValueError("Input data must be 2-dimensional")
            
        self.data = xb.astype(np.float32)  # Ensure float32 for better performance
        
        if self.verbose:
            print(f"Building NNDescent index for {len(xb)} vectors...")
            start_time = time.time()
        
        self.index = NNDescent(
            self.data,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            random_state=42,
            low_memory=True,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # Initialize the graph (important for performance)
        self.index._init_search_graph()
        
        if self.verbose:
            print(f"Index built in {time.time() - start_time:.2f} seconds")

    def search(self, xq, k):
        """
        Search the index for nearest neighbors
        
        Parameters:
        - xq: Query vectors (numpy array)
        - k: Number of neighbors to return
        
        Returns:
        - distances: Array of distances
        - indices: Array of indices
        """
        if self.index is None:
            raise RuntimeError("Index must be built before searching")
            
        if not isinstance(xq, np.ndarray):
            raise ValueError("Query data must be a numpy array")
            
        if len(xq.shape) != 2:
            raise ValueError("Query data must be 2-dimensional")
            
        if xq.shape[1] != self.data.shape[1]:
            raise ValueError("Query dimension must match index dimension")
            
        xq = xq.astype(np.float32)  # Ensure float32 for better performance
        
        if self.verbose:
            print(f"Searching for {k} neighbors in {len(xq)} queries...")
            start_time = time.time()
        
        # Query the index
        indices, distances = self.index.query(xq, k=k)
        
        if self.verbose:
            print(f"Search completed in {time.time() - start_time:.2f} seconds")
        
        return distances, indices

    def save(self, filename):
        """Save the index to disk"""
        if self.index is None:
            raise RuntimeError("Index must be built before saving")
        self.index.save(filename)

    def load(self, filename):
        """Load the index from disk"""
        self.index = NNDescent.load(filename)
        self.data = self.index._raw_data