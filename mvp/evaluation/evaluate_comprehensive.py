"""
Comprehensive Evaluation Script for Transaction Categorization System

Generates:
- Macro and per-class F1 scores
- Confusion matrix visualization
- Detailed classification report
- Performance metrics report
- Reproducibility documentation
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Import the indexer and taxonomy loader from api
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from mvp.src.api import INDEXER, load_taxonomy, choose_category_from_neighbors


def load_test_data(csv_path: Path, max_samples: int = 5000) -> Tuple[List[str], List[str]]:
    """Load test dataset with descriptions and true labels."""
    df = pd.read_csv(csv_path)
    # Sample if too large to avoid memory issues
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"Sampled {max_samples} rows from {len(pd.read_csv(csv_path))} total rows")
    texts = df["description"].astype(str).tolist()
    # Ensure we get the category column, not description
    if "category" in df.columns:
        labels = df["category"].astype(str).tolist()
    elif "category_id" in df.columns:
        labels = df["category_id"].astype(str).tolist()
    else:
        raise ValueError(f"CSV must have 'category' or 'category_id' column. Found: {df.columns.tolist()}")
    return texts, labels


def predict_batch(texts: List[str]) -> List[str]:
    """Predict categories for a batch of transactions using the indexer."""
    predictions = []
    for text in texts:
        neighbors = INDEXER.query(text, k=6)
        predicted_label, _ = choose_category_from_neighbors(neighbors)
        if predicted_label is None:
            predicted_label = "UNKNOWN"
        predictions.append(predicted_label)
    return predictions


def evaluate_model(
    test_csv: Path,
    output_dir: Path,
    split_name: str = "test"
) -> Dict:
    """Run comprehensive evaluation and generate all reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading test data from {test_csv}...")
    texts, y_true = load_test_data(test_csv, max_samples=5000)
    
    print(f"Running predictions on {len(texts)} samples...")
    start_time = time.time()
    y_pred = predict_batch(texts)
    inference_time = time.time() - start_time
    
    # Get taxonomy categories to limit confusion matrix size
    taxonomy = load_taxonomy()
    taxonomy_categories = {cat["id"] for cat in taxonomy.get("categories", [])}
    taxonomy_categories.add("UNKNOWN")
    
    # Normalize labels to taxonomy categories (map any non-taxonomy labels to UNKNOWN)
    y_true_normalized = [label if label in taxonomy_categories else "UNKNOWN" for label in y_true]
    y_pred_normalized = [label if label in taxonomy_categories else "UNKNOWN" for label in y_pred]
    
    # Get all unique labels that appear in the data (only from taxonomy)
    all_labels = sorted(taxonomy_categories & (set(y_true_normalized) | set(y_pred_normalized)))
    
    if not all_labels:
        print("ERROR: No valid taxonomy categories found in predictions!")
        return {}
    
    # Use normalized data for metrics
    y_true, y_pred = y_true_normalized, y_pred_normalized
    
    # Calculate metrics
    print("Computing metrics...")
    macro_f1 = f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=all_labels, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=all_labels, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )
    
    # Classification report
    report = classification_report(
        y_true, y_pred, labels=all_labels, output_dict=True, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    # Generate confusion matrix visualization
    print("Generating confusion matrix visualization...")
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format='d')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(f'Confusion Matrix - {split_name.upper()} Set', fontsize=16, pad=20)
    plt.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{split_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save per-class metrics
    per_class_metrics = {}
    for i, label in enumerate(all_labels):
        per_class_metrics[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i])
        }
    
    with open(output_dir / f"per_class_metrics_{split_name}.json", "w") as f:
        json.dump(per_class_metrics, f, indent=2)
    
    # Save overall metrics
    overall_metrics = {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(report.get("accuracy", 0.0)),
        "total_samples": len(y_true),
        "inference_time_seconds": float(inference_time),
        "throughput_samples_per_second": len(texts) / inference_time if inference_time > 0 else 0,
        "avg_latency_ms": (inference_time / len(texts) * 1000) if len(texts) > 0 else 0
    }
    
    with open(output_dir / f"overall_metrics_{split_name}.json", "w") as f:
        json.dump(overall_metrics, f, indent=2)
    
    # Save full classification report
    with open(output_dir / f"classification_report_{split_name}.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown report
    generate_markdown_report(
        output_dir / f"evaluation_report_{split_name}.md",
        overall_metrics,
        per_class_metrics,
        all_labels,
        split_name
    )
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS - {split_name.upper()} SET")
    print(f"{'='*60}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Total Samples: {len(y_true)}")
    print(f"Inference Time: {inference_time:.2f}s")
    print(f"Throughput: {overall_metrics['throughput_samples_per_second']:.2f} samples/sec")
    print(f"Average Latency: {overall_metrics['avg_latency_ms']:.2f} ms")
    print(f"{'='*60}\n")
    
    if macro_f1 < 0.90:
        print("[WARN] WARNING: Macro F1 score is below 0.90 benchmark requirement!")
    else:
        print("[PASS] Macro F1 score meets 0.90 benchmark requirement!")
    
    return {
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,
        "labels": all_labels,
        "macro_f1": macro_f1
    }


def generate_markdown_report(
    output_path: Path,
    overall_metrics: Dict,
    per_class_metrics: Dict,
    labels: List[str],
    split_name: str
):
    """Generate comprehensive markdown evaluation report."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Transaction Categorization Evaluation Report - {split_name.upper()}\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Macro F1 Score**: {overall_metrics['macro_f1']:.4f}\n")
        f.write(f"- **Micro F1 Score**: {overall_metrics['micro_f1']:.4f}\n")
        f.write(f"- **Weighted F1 Score**: {overall_metrics['weighted_f1']:.4f}\n")
        f.write(f"- **Accuracy**: {overall_metrics['accuracy']:.4f}\n")
        f.write(f"- **Total Samples**: {overall_metrics['total_samples']}\n\n")
        
        if overall_metrics['macro_f1'] >= 0.90:
            f.write("[PASS] **Benchmark Status**: Meets requirement (F1 >= 0.90)\n\n")
        else:
            f.write("[WARN] **Benchmark Status**: Below requirement (F1 < 0.90)\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Inference Time**: {overall_metrics['inference_time_seconds']:.2f} seconds\n")
        f.write(f"- **Throughput**: {overall_metrics['throughput_samples_per_second']:.2f} samples/second\n")
        f.write(f"- **Average Latency**: {overall_metrics['avg_latency_ms']:.2f} ms per sample\n\n")
        
        f.write("## Per-Class Metrics\n\n")
        f.write("| Category | Precision | Recall | F1 Score | Support |\n")
        f.write("|----------|-----------|--------|----------|---------|\n")
        
        for label in labels:
            metrics = per_class_metrics[label]
            f.write(f"| {label} | {metrics['precision']:.4f} | "
                   f"{metrics['recall']:.4f} | {metrics['f1_score']:.4f} | "
                   f"{metrics['support']} |\n")
        
        f.write("\n## Confusion Matrix\n\n")
        f.write(f"See `confusion_matrix_{split_name}.png` for detailed visualization.\n\n")
        
        f.write("## Reproducibility\n\n")
        f.write("To reproduce these results:\n\n")
        f.write("1. Ensure the index is built with the same dataset\n")
        f.write("2. Run: `python -m mvp.evaluation.evaluate_comprehensive`\n")
        f.write("3. Results will be saved in `mvp/evaluation/` directory\n\n")


def main():
    """Main evaluation pipeline."""
    # Determine paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir.parent / "data"
    eval_dir = base_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for test split, otherwise use full dataset
    test_csv = data_dir / "transactions.csv"
    if not test_csv.exists():
        # Try generated test split
        test_csv = base_dir / "data" / "generated" / "test.csv"
        if not test_csv.exists():
            print(f"ERROR: No test data found. Expected one of:")
            print(f"  - {data_dir / 'transactions.csv'}")
            print(f"  - {base_dir / 'data' / 'generated' / 'test.csv'}")
            return
    
    print("="*60)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("="*60)
    print(f"Test data: {test_csv}")
    print(f"Output directory: {eval_dir}")
    print()
    
    # Ensure index is built
    if not INDEXER.index:
        print("Building index from transactions.csv...")
        INDEXER.load_documents_from_csv()
        INDEXER.build_index()
        print(f"Index built with {len(INDEXER.docs)} documents\n")
    
    # Run evaluation
    results = evaluate_model(test_csv, eval_dir, split_name="test")
    
    print(f"\n[SUCCESS] Evaluation complete! Results saved to: {eval_dir}")
    print(f"   - confusion_matrix_test.png")
    print(f"   - per_class_metrics_test.json")
    print(f"   - overall_metrics_test.json")
    print(f"   - classification_report_test.json")
    print(f"   - evaluation_report_test.md")


if __name__ == "__main__":
    main()

