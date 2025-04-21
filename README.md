# FAISS_4way_index-SceureNNDescent
This project is designed to benchmark privacy-preserving vector search algorithms, with a focus on FAISS-based vector search, NNDescent, and various encryption techniques. The core aim is to evaluate how different vector search algorithms perform under encryption.

To run the project, you'll need the following Python libraries:

faiss-cpu or faiss-gpu

PySEAL, TenSEAL, or PySyft for encryption libraries

numpy, scipy, matplotlib for data handling and visualization

pandas for data storage and results analysis

You can install the required dependencies using pip:

pip install -r requirements.txt

Deep1B Dataset: You will need to download the Deep1B dataset to run the benchmarks. The dataset is available from the official source.

Usage
Running the Benchmark
To run the benchmark and compare the performance of different search algorithms and encryption methods, use the following command:

python benchmark.py --index_type <index_type> --encryption_type <encryption_type> --nprobe <nprobe_value>

<index_type> can be Flat, IVF, HNSW, or LSH, corresponding to the FAISS index type.

<encryption_type> can be one of the encryption methods, such as HE, SE, OPE, DP, or None for no encryption.

<nprobe_value> is the number of probes to use for the IVF index (if applicable).

Output
The benchmark script generates CSV files and visual charts with the following metrics:

Search Time: The average time taken to perform the Top-k search.

Recall@k: The percentage of times the correct result appears in the Top-k results.

Query Time: Time taken per query for encryption and search.

Structure
The project is organized into several key components:

benchmark.py: Main script for running the benchmarks. It allows you to configure the index type, encryption method, and various parameters for the search process.

faiss_utils.py: Contains helper functions for initializing and managing FAISS indexes.

encryption_utils.py: Implements the encryption algorithms used in the project (e.g., Fully Homomorphic Encryption, Secret Sharing).

data_utils.py: Functions for loading and managing the Deep1B dataset.

config.py: Configuration file for setting up paths, parameters, and dataset options.

License
This project is licensed under the MIT License - see the LICENSE file for details.