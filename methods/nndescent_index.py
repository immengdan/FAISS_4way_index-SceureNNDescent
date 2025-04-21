from pynndescent import NNDescent
import os

class NNDescentIndexer:
    def __init__(self, n_neighbors=10, n_jobs=4):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs


        os.environ["OMP_NUM_THREADS"] = str(n_jobs)
        os.environ["MKL_NUM_THREADS"] = str(n_jobs)

    def build(self, xb):
        self.index = NNDescent(
            xb,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            random_state=42,
            low_memory=True,     
            n_jobs=self.n_jobs
        )

    def search(self, xq, k):
        indices, _ = self.index.query(xq, k=k)
        return indices
