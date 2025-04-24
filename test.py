#if you are facing some unknow problem you can try to test here
import os
from utils.metrics import evaluate_results
from utils.data_loader import load_fvecs_fbin
from methods.faiss_index import FAISSIndexer
from methods.nndescent_index import NNDescentIndexer 

base_path = os.path.join(os.path.dirname(__file__), "data/base.10M.fbin")
query_path = os.path.join(os.path.dirname(__file__), "data/query.public.10K.fbin")
dim=96
nb = 10_000
nq = 100
k=10
xb = load_fvecs_fbin(base_path, count=nb, dim=dim)
xq = load_fvecs_fbin(query_path, count=nq, dim=dim)

flat_index = FAISSIndexer(dim, "flat")
flat_index.add_with_progress(xb)
true_D, true_I = flat_index.search(xq, k)

recall_dict = {}

def run_faiss_baseline():
    indexer = FAISSIndexer(dim, "flat")
    indexer.add_with_progress(xb)
    _, I = indexer.search(xq, k)
    recall = evaluate_results(true_I, I, k)
    print("FAISS recall:", recall)

def run_nndescent_baseline():
    nndescent = NNDescentIndexer(n_neighbors=k)
    nndescent.build(xb)
    indices, distances = nndescent.search(xq, k)
    print("Indices shape:", indices.shape)
    print("Indices sample:", indices[0])
    recall = evaluate_results(true_I, indices, k)
    print("True_I shape:", true_I.shape)
    print("True_I sample:", true_I[0]) 
    print("NNDescent recall:", recall)

run_faiss_baseline()
run_nndescent_baseline()