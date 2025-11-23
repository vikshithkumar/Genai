# Transaction Categorization Evaluation Report - TEST

Generated: 2025-11-22 11:12:50

## Executive Summary

- **Macro F1 Score**: 1.0000
- **Micro F1 Score**: 1.0000
- **Weighted F1 Score**: 1.0000
- **Accuracy**: 1.0000
- **Total Samples**: 5000

[PASS] **Benchmark Status**: Meets requirement (F1 >= 0.90)

## Performance Metrics

- **Inference Time**: 196.63 seconds
- **Throughput**: 25.43 samples/second
- **Average Latency**: 39.33 ms per sample

## Per-Class Metrics

| Category | Precision | Recall | F1 Score | Support |
|----------|-----------|--------|----------|---------|
| BANKING | 1.0000 | 1.0000 | 1.0000 | 362 |
| EDUCATION | 1.0000 | 1.0000 | 1.0000 | 363 |
| ENTERTAINMENT | 1.0000 | 1.0000 | 1.0000 | 365 |
| FUEL | 1.0000 | 1.0000 | 1.0000 | 382 |
| GROCERIES | 1.0000 | 1.0000 | 1.0000 | 377 |
| HEALTH | 1.0000 | 1.0000 | 1.0000 | 353 |
| INCOME | 1.0000 | 1.0000 | 1.0000 | 361 |
| RENT | 1.0000 | 1.0000 | 1.0000 | 330 |
| RESTAURANTS | 1.0000 | 1.0000 | 1.0000 | 358 |
| SHOPPING | 1.0000 | 1.0000 | 1.0000 | 338 |
| SUBSCRIPTIONS | 1.0000 | 1.0000 | 1.0000 | 347 |
| TRANSPORT | 1.0000 | 1.0000 | 1.0000 | 367 |
| TRAVEL | 1.0000 | 1.0000 | 1.0000 | 355 |
| UTILITIES | 1.0000 | 1.0000 | 1.0000 | 342 |

## Confusion Matrix

See `confusion_matrix_test.png` for detailed visualization.

## Reproducibility

To reproduce these results:

1. Ensure the index is built with the same dataset
2. Run: `python -m mvp.evaluation.evaluate_comprehensive`
3. Results will be saved in `mvp/evaluation/` directory

