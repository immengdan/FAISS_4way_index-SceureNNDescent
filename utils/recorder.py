import os
import json
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np 

class BenchmarkRecorder:
    def __init__(self, experiment_name="benchmark"):
        from datetime import datetime
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results/{experiment_name}_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.results = {
            "metadata": {},
            "runs": []
        }

    def record_metadata(self, dim, base_size, query_size):
        self.results["metadata"] = {
            "dimension": dim,
            "base_size": base_size,
            "query_size": query_size
        }

    def record_run(self, run_name, params, results, notes=""):
        from datetime import datetime
        self.results["runs"].append({
            "run_name": run_name,
            "params": params,
            "results": results,
            "notes": notes,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def plot_metrics(self):
        def ensure_scalar(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                return float(np.mean(val)) 
            return float(val)

        labels = [run["run_name"] for run in self.results["runs"]]
        recalls = [ensure_scalar(run["results"].get("recall", 0)) for run in self.results["runs"]]
        precisions = [ensure_scalar(run["results"].get("precision", 0)) for run in self.results["runs"]]
        times = [ensure_scalar(run["results"].get("query_time", 0)) for run in self.results["runs"]]
        distances = [ensure_scalar(run["results"].get("mean_distance", 0)) for run in self.results["runs"]]

        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        ax[0, 0].barh(labels, recalls, color='skyblue')
        ax[0, 0].set_title("Recall@k")
        ax[0, 0].invert_yaxis()

        ax[0, 1].barh(labels, precisions, color='lightgreen')
        ax[0, 1].set_title("Precision@k")
        ax[0, 1].invert_yaxis()

        ax[1, 0].barh(labels, times, color='salmon')
        ax[1, 0].set_title("Query Time (s)")
        ax[1, 0].invert_yaxis()

        ax[1, 1].barh(labels, distances, color='lightcoral')
        ax[1, 1].set_title("Mean Distance")
        ax[1, 1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "benchmark_plot.png"))
        plt.close()

    def save_all(self):
        def convert_to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(x) for x in obj]
            return obj

        serializable_results = convert_to_serializable(self.results)

        json_path = os.path.join(self.results_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        pkl_path = os.path.join(self.results_dir, "results.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(self.results, f)

        df = pd.DataFrame([
            {
                "run_name": r["run_name"],
                "query_time": float(r["results"].get("query_time", 0)),
               "recall": r["results"].get("recall", []),
                "precision": float(r["results"].get("precision", 0)),
                "mean_distance": float(r["results"].get("mean_distance", 0)),
                "index_time": float(r["results"].get("index_time", 0)),
                "index_memory_MB": float(r["results"].get("index_memory_MB", 0)),
               "index_size_MB": float(r["results"].get("index_size_MB", 0)),
                "method": r["params"].get("method", ""),
                "encrypted": r["params"].get("encrypted", False),
                "encryption": r["params"].get("encryption", "None"),
                "notes": r["notes"],
                "timestamp": r["timestamp"]
            }
            for r in self.results["runs"]
        ])
        df.to_csv(os.path.join(self.results_dir, "results.csv"), index=False)
        self.plot_metrics()

        print(f"Results saved to: {self.results_dir}")

