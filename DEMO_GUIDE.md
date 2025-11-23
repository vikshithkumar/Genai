# Demo Guide: AI Financial Transaction Categorization System

This guide provides step-by-step instructions for demonstrating all key features of the system.

## Prerequisites

1. **Start the API server**:
   ```bash
   cd ai-financial-tx-categorization-mvp
   uvicorn mvp.src.api:app --reload
   ```

2. **Verify server is running**: Open `http://localhost:8000` in browser

---

## Demo Flow

### 1. Pipeline Execution Demo

**Objective**: Show the complete pipeline from data to predictions

#### Step 1.1: Generate Synthetic Dataset
```bash
# Generate 20,000 transaction samples
python -m mvp.src.generate_synthetic_dataset --rows 20000 --out data/transactions.csv
```

**What to show**:
- Dataset generation process
- Output file location (`data/transactions.csv`)
- Sample rows from the CSV

#### Step 1.2: Build Index (Automatic on Startup)
The index is automatically built when the API server starts. You can also manually rebuild:

```bash
# Via API
curl -X POST http://localhost:8000/rebuild_index
```

**What to show**:
- Index building process (logs in terminal)
- Index metadata: `GET http://localhost:8000/index_meta`
- Number of documents indexed

#### Step 1.3: Verify Pipeline
```bash
# Check index status
curl http://localhost:8000/index_meta
```

**Expected Response**:
```json
{
  "model": "all-MiniLM-L6-v2",
  "count": 20000
}
```

---

### 2. Evaluation Demo

**Objective**: Show comprehensive evaluation metrics

#### Step 2.1: Run Comprehensive Evaluation
```bash
python -m mvp.evaluation.evaluate_comprehensive
```

**What to show**:
- Evaluation process running
- Metrics being calculated
- Output files generated

#### Step 2.2: Review Evaluation Results

**Files Generated** (in `mvp/evaluation/`):
- `confusion_matrix_test.png` - Visual confusion matrix
- `per_class_metrics_test.json` - Per-category metrics
- `overall_metrics_test.json` - Aggregate metrics
- `classification_report_test.json` - Full sklearn report
- `evaluation_report_test.md` - Human-readable summary

**Key Metrics to Highlight**:
```bash
# View overall metrics
cat mvp/evaluation/overall_metrics_test.json

# View evaluation report
cat mvp/evaluation/evaluation_report_test.md
```

**What to show**:
- **Macro F1 Score**: Should be ≥ 0.90 (typically 1.0)
- **Accuracy**: Should be high (typically 1.0)
- **Per-class metrics**: All categories performing well
- **Confusion matrix**: Visual representation of performance

#### Step 2.3: Performance Benchmarking (Optional)
```bash
python -m mvp.evaluation.benchmark_performance
```

**What to show**:
- Latency metrics (P50, P95, P99)
- Throughput (samples/second)
- Performance benchmarks JSON

---

### 3. Sample Predictions with Confidence Demo

**Objective**: Demonstrate prediction accuracy and confidence scores

#### Step 3.1: Single Prediction via API
```bash
# Example 1: Grocery transaction
curl -X POST http://localhost:8000/predict \
  -F "transaction=Zepto grocery order 1234"

# Example 2: Restaurant transaction
curl -X POST http://localhost:8000/predict \
  -F "transaction=Swiggy food delivery order 5678"

# Example 3: Shopping transaction
curl -X POST http://localhost:8000/predict \
  -F "transaction=Amazon India purchase order 9012"
```

**Expected Response Format**:
```json
{
  "description": "Zepto grocery order 1234",
  "predicted_category_id": "GROCERIES",
  "predicted_category_name": "Groceries",
  "confidence": 0.95,
  "is_low_confidence": false,
  "explanations": [
    {
      "description": "Zepto grocery order 4590",
      "category": "GROCERIES",
      "similarity": 0.92
    }
  ],
  "keyword_matches": ["zepto"],
  "rationale": "Prediction GROCERIES (confidence 0.95) based on nearest neighbors",
  "feature_importance": {
    "keywords": {...},
    "neighbors": {...}
  }
}
```

**What to show**:
- ✅ **High confidence predictions** (≥ 0.90)
- ✅ **Category accuracy** (correct predictions)
- ✅ **Explainability** (nearest neighbors, keyword matches)
- ✅ **Feature importance** (keyword weights, neighbor contributions)

#### Step 3.2: Low Confidence Prediction Demo
```bash
# Test with ambiguous transaction
curl -X POST http://localhost:8000/predict \
  -F "transaction=Unknown merchant payment 9999"
```

**What to show**:
- Low confidence flag (`is_low_confidence: true`)
- Confidence score below threshold
- How system handles uncertainty

#### Step 3.3: Batch Prediction Demo
```bash
# Create a test CSV file
cat > test_batch.csv << EOF
description
Amazon Prime subscription
Zepto grocery delivery
Uber ride to airport
Netflix monthly payment
EOF

# Run batch prediction
curl -X POST http://localhost:8000/predict_batch \
  -F "file=@test_batch.csv"
```

**What to show**:
- Multiple predictions at once
- Batch processing efficiency
- Results for all transactions

#### Step 3.4: UI Demo (Recommended)
1. Open `http://localhost:8000/ui/index.html` in browser
2. Enter transactions in the "Single Transaction" section
3. Click "Predict" button
4. Show:
   - Predicted category
   - Confidence score
   - Nearest neighbors
   - Keyword matches
   - Feature importance visualization
   - Rationale

**Sample Transactions to Test**:
- `"Zepto grocery order 1234"` → GROCERIES (high confidence)
- `"Swiggy food delivery"` → RESTAURANTS (high confidence)
- `"Amazon Prime Video"` → ENTERTAINMENT (high confidence)
- `"Uber ride to airport"` → TRANSPORT (high confidence)
- `"HDFC Bank ATM withdrawal"` → BANKING (high confidence)

---

### 4. Taxonomy Modification Demo

**Objective**: Show how to customize categories without code changes

#### Step 4.1: View Current Taxonomy
```bash
# Get current taxonomy
curl http://localhost:8000/taxonomy

# Or download it
curl http://localhost:8000/download_taxonomy -o current_taxonomy.json
```

**What to show**:
- Current categories
- Category structure (id, name, aliases)
- Configuration format

#### Step 4.2: Modify Taxonomy

**Option A: Via UI** (Easiest)
1. Go to `http://localhost:8000/ui/index.html`
2. Scroll to "Taxonomy Admin" section
3. Click "Choose File" and select a modified `taxonomy.json`
4. Click "Upload Taxonomy"
5. Click "Rebuild Index" to apply changes

**Option B: Via API**
```bash
# Upload modified taxonomy
curl -X POST http://localhost:8000/upload_taxonomy \
  -F "file=@modified_taxonomy.json"

# Rebuild index to apply changes
curl -X POST http://localhost:8000/rebuild_index
```

#### Step 4.3: Example Taxonomy Modification

**Create a modified taxonomy** (`modified_taxonomy.json`):
```json
{
  "model": "all-MiniLM-L6-v2",
  "low_confidence_threshold": 0.5,
  "categories": [
    {
      "id": "GROCERIES",
      "name": "Groceries",
      "aliases": ["zepto", "blinkit", "bigbasket", "grocery", "supermarket", "instamart"]
    },
    {
      "id": "RESTAURANTS",
      "name": "Restaurants & Cafes",
      "aliases": ["swiggy", "zomato", "restaurant", "cafe", "food delivery", "uber eats"]
    },
    {
      "id": "SHOPPING",
      "name": "Shopping",
      "aliases": ["amazon", "flipkart", "myntra", "shopping", "ecommerce", "online shopping"]
    },
    {
      "id": "NEW_CATEGORY",
      "name": "New Category Example",
      "aliases": ["newmerchant", "testmerchant"]
    }
  ]
}
```

**What to show**:
- ✅ Adding new category (`NEW_CATEGORY`)
- ✅ Adding new aliases to existing categories
- ✅ Taxonomy upload process
- ✅ Index rebuild process
- ✅ Testing predictions with new taxonomy

#### Step 4.4: Verify Taxonomy Changes
```bash
# Check updated taxonomy
curl http://localhost:8000/taxonomy

# Test prediction with new category
curl -X POST http://localhost:8000/predict \
  -F "transaction=newmerchant purchase 1234"
```

**What to show**:
- Taxonomy successfully updated
- New category recognized
- Predictions work with modified taxonomy

---

## Complete Demo Script (5-10 minutes)

### Quick Demo Flow:

1. **Introduction** (30 seconds)
   - "This is an AI-based financial transaction categorization system"
   - "It categorizes transactions like 'Zepto grocery' into categories like 'GROCERIES'"
   - "All processing happens locally - no external APIs"

2. **Pipeline Execution** (1 minute)
   - Show dataset generation: `python -m mvp.src.generate_synthetic_dataset --rows 20000`
   - Show index building (automatic on startup)
   - Show index metadata: `curl http://localhost:8000/index_meta`

3. **Evaluation** (2 minutes)
   - Run evaluation: `python -m mvp.evaluation.evaluate_comprehensive`
   - Show results:
     - Macro F1 score (≥ 0.90)
     - Confusion matrix image
     - Per-class metrics
   - Highlight: "Exceeds benchmark requirement of 0.90"

4. **Sample Predictions** (3 minutes)
   - Open UI: `http://localhost:8000/ui/index.html`
   - Make 3-4 predictions:
     - "Zepto grocery order" → GROCERIES (95% confidence)
     - "Swiggy food delivery" → RESTAURANTS (92% confidence)
     - "Amazon Prime Video" → ENTERTAINMENT (88% confidence)
   - Show explainability:
     - Nearest neighbors
     - Keyword matches
     - Feature importance
     - Rationale

5. **Taxonomy Modification** (2 minutes)
   - Show current taxonomy: `curl http://localhost:8000/taxonomy`
   - Modify taxonomy (add new category or alias)
   - Upload via UI or API
   - Rebuild index
   - Test prediction with new taxonomy
   - Show: "No code changes needed - just JSON configuration"

6. **Bonus Features** (1 minute)
   - Show continuous learning: Submit correction → Auto-retrain
   - Show cost analysis: `curl http://localhost:8000/cost_analysis?scenario=large`
   - Show batch prediction capability

---

## Demo Checklist

Before the demo, ensure:

- [ ] API server is running (`uvicorn mvp.src.api:app --reload`)
- [ ] Dataset is generated (`data/transactions.csv` exists)
- [ ] Index is built (check `GET /index_meta`)
- [ ] Evaluation has been run (results in `mvp/evaluation/`)
- [ ] UI is accessible (`http://localhost:8000/ui/index.html`)
- [ ] Sample transactions ready for testing
- [ ] Modified taxonomy file ready (for modification demo)

---

## Troubleshooting

### Issue: "No transactions CSV found"
**Solution**: Generate dataset first:
```bash
python -m mvp.src.generate_synthetic_dataset --rows 20000
```

### Issue: "Index not built"
**Solution**: Rebuild index:
```bash
curl -X POST http://localhost:8000/rebuild_index
```

### Issue: "Evaluation fails"
**Solution**: Ensure test data exists:
```bash
# Check if test.csv exists
ls mvp/data/generated/test.csv

# If not, generate it
python -m mvp.src.generate_synthetic_dataset --rows 20000
```

### Issue: "UI not loading"
**Solution**: Check API server is running and UI directory exists:
```bash
# Verify UI directory
ls mvp/ui/

# Check server logs for errors
```

---

## Key Talking Points

1. **Autonomy**: "No external API dependencies - everything runs locally"
2. **Accuracy**: "Macro F1 score of 1.0, exceeding the 0.90 benchmark"
3. **Customization**: "Easy taxonomy updates via JSON - no code changes"
4. **Explainability**: "Every prediction includes explanations - nearest neighbors, keywords, feature importance"
5. **Continuous Learning**: "System learns from corrections automatically"
6. **Cost Savings**: "Significant ROI compared to third-party APIs, especially at scale"
7. **Robustness**: "Handles noisy transaction strings with misspellings and IDs"

---

## Video Recording Tips

If recording the demo:

1. **Screen Layout**:
   - Terminal (left): Show commands and API responses
   - Browser (right): Show UI interactions
   - Or use split screen

2. **Highlight**:
   - Command execution
   - API responses (JSON)
   - UI interactions
   - Results/metrics

3. **Narration**:
   - Explain what you're doing
   - Highlight key features
   - Show confidence scores
   - Demonstrate explainability

---

*Last Updated: 2025-01-XX*

