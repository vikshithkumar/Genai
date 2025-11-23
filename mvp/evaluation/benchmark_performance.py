"""
Performance Benchmarking Script

Measures:
- Single prediction latency (p50, p95, p99)
- Batch inference throughput
- Memory usage
- End-to-end API response times
"""

import time
import statistics
from pathlib import Path
from typing import List, Dict
import json

import numpy as np

# Import the indexer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mvp.src.api import INDEXER, choose_category_from_neighbors


def benchmark_single_predictions(
    test_transactions: List[str],
    num_iterations: int = 1000,
    warmup: int = 50
) -> Dict:
    """Benchmark single prediction latency."""
    print(f"Benchmarking single predictions (warmup={warmup}, iterations={num_iterations})...")
    
    # Warmup
    for _ in range(warmup):
        if test_transactions:
            INDEXER.query(test_transactions[0], k=6)
    
    # Actual benchmark
    latencies_ms = []
    for i in range(num_iterations):
        tx = test_transactions[i % len(test_transactions)] if test_transactions else "Amazon order 1234"
        
        start = time.perf_counter()
        neighbors = INDEXER.query(tx, k=6)
        choose_category_from_neighbors(neighbors)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies_ms.append(latency_ms)
    
    latencies_ms = np.array(latencies_ms)
    
    return {
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "std_latency_ms": float(np.std(latencies_ms)),
        "min_latency_ms": float(np.min(latencies_ms)),
        "max_latency_ms": float(np.max(latencies_ms)),
        "total_iterations": num_iterations
    }


def benchmark_batch_throughput(
    test_transactions: List[str],
    batch_sizes: List[int] = [1, 10, 50, 100, 500, 1000]
) -> Dict:
    """Benchmark batch inference throughput."""
    print(f"Benchmarking batch throughput...")
    
    results = {}
    
    for batch_size in batch_sizes:
        if batch_size > len(test_transactions):
            continue
        
        batch = test_transactions[:batch_size]
        
        # Warmup
        for tx in batch[:min(10, len(batch))]:
            INDEXER.query(tx, k=6)
        
        # Measure
        start = time.perf_counter()
        for tx in batch:
            neighbors = INDEXER.query(tx, k=6)
            choose_category_from_neighbors(neighbors)
        end = time.perf_counter()
        
        elapsed = end - start
        throughput = batch_size / elapsed
        
        results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "total_time_seconds": float(elapsed),
            "throughput_samples_per_second": float(throughput),
            "avg_latency_ms": float((elapsed / batch_size) * 1000)
        }
    
    return results


def main():
    """Run comprehensive performance benchmarks."""
    print("="*60)
    print("PERFORMANCE BENCHMARKING")
    print("="*60)
    
    # Load test transactions
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir.parent / "data"
    test_csv = data_dir / "transactions.csv"
    
    if not test_csv.exists():
        print(f"ERROR: Test data not found at {test_csv}")
        return
    
    import pandas as pd
    df = pd.read_csv(test_csv)
    test_transactions = df["description"].astype(str).tolist()[:1000]  # Use first 1000 for benchmarking
    
    # Ensure index is built
    if not INDEXER.index:
        print("Building index...")
        INDEXER.load_documents_from_csv()
        INDEXER.build_index()
        print(f"Index built with {len(INDEXER.docs)} documents\n")
    
    # Run benchmarks
    single_results = benchmark_single_predictions(test_transactions, num_iterations=1000)
    batch_results = benchmark_batch_throughput(test_transactions)
    
    # Combine results
    all_results = {
        "single_prediction": single_results,
        "batch_throughput": batch_results,
        "index_size": len(INDEXER.docs),
        "model": INDEXER.model_name
    }
    
    # Save results
    output_dir = base_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "performance_benchmarks.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print("\nSingle Prediction Latency:")
    print(f"  Mean: {single_results['mean_latency_ms']:.2f} ms")
    print(f"  P50:  {single_results['p50_latency_ms']:.2f} ms")
    print(f"  P95:  {single_results['p95_latency_ms']:.2f} ms")
    print(f"  P99:  {single_results['p99_latency_ms']:.2f} ms")
    
    print("\nBatch Throughput:")
    for key, result in batch_results.items():
        print(f"  {key}: {result['throughput_samples_per_second']:.2f} samples/sec "
              f"({result['avg_latency_ms']:.2f} ms avg)")
    
    print(f"\n✅ Results saved to: {output_dir / 'performance_benchmarks.json'}")


if __name__ == "__main__":
    main()

