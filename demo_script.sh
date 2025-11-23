#!/bin/bash
# Demo Script for AI Transaction Categorization System
# This script automates the demo flow

set -e  # Exit on error

echo "=========================================="
echo "AI Transaction Categorization Demo"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"

# Check if server is running
echo -e "${BLUE}Step 1: Checking API server...${NC}"
if curl -s "$API_URL" > /dev/null; then
    echo -e "${GREEN}✓ API server is running${NC}"
else
    echo -e "${YELLOW}⚠ API server not running. Please start it with:${NC}"
    echo "  uvicorn mvp.src.api:app --reload"
    exit 1
fi
echo ""

# 1. Pipeline Execution Demo
echo -e "${BLUE}Step 2: Pipeline Execution Demo${NC}"
echo "----------------------------------------"

# Check if dataset exists
if [ -f "data/transactions.csv" ]; then
    echo -e "${GREEN}✓ Dataset exists (data/transactions.csv)${NC}"
    echo "  Row count: $(wc -l < data/transactions.csv)"
else
    echo -e "${YELLOW}⚠ Generating dataset...${NC}"
    python -m mvp.src.generate_synthetic_dataset --rows 20000 --out data/transactions.csv
    echo -e "${GREEN}✓ Dataset generated${NC}"
fi

# Check index status
echo ""
echo "Checking index status..."
INDEX_META=$(curl -s "$API_URL/index_meta")
echo "$INDEX_META" | python -m json.tool
echo ""

# 2. Evaluation Demo
echo -e "${BLUE}Step 3: Evaluation Demo${NC}"
echo "----------------------------------------"
echo "Running comprehensive evaluation..."
echo ""

if [ -f "mvp/evaluation/overall_metrics_test.json" ]; then
    echo -e "${GREEN}✓ Evaluation results found${NC}"
    echo ""
    echo "Overall Metrics:"
    cat mvp/evaluation/overall_metrics_test.json | python -m json.tool
    echo ""
    echo "Macro F1 Score:"
    cat mvp/evaluation/overall_metrics_test.json | python -c "import sys, json; print(json.load(sys.stdin)['macro_f1'])"
else
    echo -e "${YELLOW}⚠ Running evaluation...${NC}"
    python -m mvp.evaluation.evaluate_comprehensive
    echo -e "${GREEN}✓ Evaluation complete${NC}"
fi
echo ""

# 3. Sample Predictions Demo
echo -e "${BLUE}Step 4: Sample Predictions with Confidence${NC}"
echo "----------------------------------------"

# Test transactions
declare -a transactions=(
    "Zepto grocery order 1234"
    "Swiggy food delivery order 5678"
    "Amazon Prime Video subscription"
    "Uber ride to airport"
    "Netflix monthly payment"
)

for tx in "${transactions[@]}"; do
    echo ""
    echo "Testing: '$tx'"
    echo "---"
    curl -s -X POST "$API_URL/predict" \
        -F "transaction=$tx" | \
        python -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Category: {data['predicted_category_name']}\")
print(f\"Confidence: {data['confidence']:.2%}\")
print(f\"Low Confidence: {data['is_low_confidence']}\")
print(f\"Keywords: {', '.join(data.get('keyword_matches', []))}\")
"
done
echo ""

# 4. Taxonomy Modification Demo
echo -e "${BLUE}Step 5: Taxonomy Modification Demo${NC}"
echo "----------------------------------------"

# Get current taxonomy
echo "Current taxonomy:"
curl -s "$API_URL/taxonomy" | python -m json.tool | head -20
echo ""

# Create a modified taxonomy for demo
echo "Creating modified taxonomy (adding new alias)..."
cat > /tmp/demo_taxonomy.json << 'EOF'
{
  "model": "all-MiniLM-L6-v2",
  "low_confidence_threshold": 0.5,
  "categories": [
    {
      "id": "GROCERIES",
      "name": "Groceries",
      "aliases": ["zepto", "blinkit", "bigbasket", "grocery", "supermarket", "instamart", "demo_alias"]
    },
    {
      "id": "RESTAURANTS",
      "name": "Restaurants & Cafes",
      "aliases": ["swiggy", "zomato", "restaurant", "cafe", "food delivery"]
    },
    {
      "id": "SHOPPING",
      "name": "Shopping",
      "aliases": ["amazon", "flipkart", "myntra", "shopping", "ecommerce"]
    }
  ]
}
EOF

echo -e "${YELLOW}⚠ Uploading modified taxonomy...${NC}"
curl -s -X POST "$API_URL/upload_taxonomy" \
    -F "file=@/tmp/demo_taxonomy.json" | python -m json.tool

echo ""
echo -e "${YELLOW}⚠ Rebuilding index with new taxonomy...${NC}"
curl -s -X POST "$API_URL/rebuild_index" | python -m json.tool

echo ""
echo -e "${GREEN}✓ Taxonomy updated and index rebuilt${NC}"

# Test with new taxonomy
echo ""
echo "Testing prediction with updated taxonomy:"
curl -s -X POST "$API_URL/predict" \
    -F "transaction=demo_alias purchase 9999" | \
    python -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Category: {data['predicted_category_name']}\")
print(f\"Confidence: {data['confidence']:.2%}\")
"

echo ""
echo "=========================================="
echo -e "${GREEN}Demo Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Open UI: http://localhost:8000/ui/index.html"
echo "2. Try more predictions in the UI"
echo "3. View evaluation results: mvp/evaluation/"
echo "4. Check cost analysis: $API_URL/cost_analysis?scenario=large"

