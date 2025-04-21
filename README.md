# FAISS_4way_index-SceureNNDescent
This project is designed to benchmark privacy-preserving vector search algorithms, with a focus on FAISS-based vector search, NNDescent, and various encryption techniques. The core aim is to evaluate how different vector search algorithms perform under encryption.

## Requirements
To run the project, you'll need the following Python libraries:
- **faiss-cpu** or **faiss-gpu** (for FAISS-based vector search)
- **PySEAL**, **TenSEAL**, or **PySyft** (for encryption libraries)
- **numpy**, **scipy**, **matplotlib** (for data handling and visualization)
- **pandas** (for data storage and results analysis)
  
You can install the required dependencies using pip:
 ```
pip install -r requirements.txt
 ```

## Data
Deep1B Dataset: You will need to download the [Deep1B dataset](https://learning2hash.github.io/publications/yandexdeep1B/![image](https://github.com/user-attachments/assets/038c128b-d306-45bd-a154-93844f6cffba)
) to run the benchmarks. The dataset is available from the official source.
open URL - find https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search#13h2 
DEEP-1B Documentation
Base vector: Load the subset of 10M embeddings as your database vectors.
Query vector:Load the query set (10K embeddings) for your queries.
Vector Dimension: 96

## Usage
Running the Benchmark
To run the benchmark and compare the performance of different search algorithms and encryption methods, use the following command:
 ```
python benchmark.py
 ```

## Output
The benchmark script generates CSV files and visual charts with the following metrics:
-Search Time: The average time taken to perform the Top-k search.
-Recall@k: The percentage of times the correct result appears in the Top-k results.
-Query Time: Time taken per query for encryption and search.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
