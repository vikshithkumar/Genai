# Bias Mitigation & Ethical AI Considerations

## Overview

This document outlines the bias mitigation strategies and ethical considerations implemented in the AI-based financial transaction categorization system.

## Potential Sources of Bias

### 1. **Merchant/Region Bias**
- **Risk**: The system may favor certain merchants or regions if training data is imbalanced
- **Mitigation**:
  - Synthetic dataset generation ensures balanced representation across categories
  - Taxonomy aliases include diverse merchant names from multiple regions (India, US, UK)
  - No geographic or demographic information is used in classification decisions

### 2. **Category Imbalance**
- **Risk**: Some categories (e.g., "SHOPPING") may have more training examples than others
- **Mitigation**:
  - Dataset generator ensures minimum samples per category
  - Macro F1 score (rather than accuracy) is used to evaluate performance, giving equal weight to all categories
  - Per-class metrics are monitored to identify underperforming categories

### 3. **Amount-Based Bias**
- **Risk**: System might associate certain amounts with specific categories
- **Mitigation**:
  - Transaction amounts are not used as features in the model
  - Only text descriptions are used for classification
  - Amount normalization/removal in preprocessing prevents amount-based patterns

### 4. **Language/Cultural Bias**
- **Risk**: System may perform poorly on non-English or culturally-specific transaction descriptions
- **Mitigation**:
  - Taxonomy includes aliases in multiple languages (e.g., Hindi merchant names)
  - Dataset includes diverse transaction formats (UPI, NEFT, etc.)
  - No assumptions about language or cultural context in classification logic

## Implemented Safeguards

### 1. **Transparent Taxonomy**
- All category definitions and aliases are stored in `taxonomy.json`
- Admins can review and modify categories without code changes
- No hidden or implicit category associations

### 2. **Explainability**
- Every prediction includes:
  - Nearest neighbor examples (showing similar transactions)
  - Keyword matches (showing which terms influenced the decision)
  - Confidence scores (flagging uncertain predictions)
- Users can review and understand why a category was assigned

### 3. **Feedback Loop**
- Low-confidence predictions are flagged for human review
- User corrections are collected in `corrections_buffer.jsonl`
- Corrections can be used to retrain/update the model

### 4. **No Sensitive Attributes**
- The system does not use:
  - User demographics (age, gender, location)
  - Transaction amounts
  - Account balances or financial status
  - Personal identifiers

### 5. **Fair Evaluation**
- Macro F1 score ensures all categories are evaluated equally
- Per-class metrics identify categories needing improvement
- Confusion matrix reveals systematic misclassifications

## Recommendations for Production

1. **Regular Auditing**: Periodically review predictions for systematic biases
2. **Diverse Training Data**: Ensure training data represents diverse user populations
3. **User Feedback**: Actively collect and incorporate user corrections
4. **Category Review**: Regularly review taxonomy for fairness and inclusivity
5. **Performance Monitoring**: Track per-category performance metrics over time
6. **Transparency**: Clearly communicate to users that predictions are assistive and may require review

## Ethical Guidelines

- **Privacy**: No personal information is stored or transmitted
- **Transparency**: All classification logic is explainable
- **Fairness**: Equal treatment across all categories and user groups
- **Accountability**: Users can review and correct predictions
- **Responsibility**: System is designed as an assistive tool, not a replacement for human judgment

## Future Improvements

1. **Bias Detection**: Automated tools to detect systematic biases in predictions
2. **Fairness Metrics**: Additional metrics beyond F1 (e.g., demographic parity)
3. **Adversarial Testing**: Test system robustness to adversarial inputs
4. **Continuous Learning**: Incorporate user feedback into model updates
5. **Multi-language Support**: Expand support for non-English transaction descriptions

