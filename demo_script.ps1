# Demo Script for AI Transaction Categorization System (PowerShell)
# This script automates the demo flow for Windows

$ErrorActionPreference = "Stop"
$API_URL = "http://localhost:8000"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AI Transaction Categorization Demo" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check API server
Write-Host "Step 1: Checking API server..." -ForegroundColor Blue
try {
    $response = Invoke-WebRequest -Uri $API_URL -Method GET -UseBasicParsing -ErrorAction Stop
    Write-Host "✓ API server is running" -ForegroundColor Green
} catch {
    Write-Host "⚠ API server not running. Please start it with:" -ForegroundColor Yellow
    Write-Host "  uvicorn mvp.src.api:app --reload" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Step 2: Pipeline Execution Demo
Write-Host "Step 2: Pipeline Execution Demo" -ForegroundColor Blue
Write-Host "----------------------------------------" -ForegroundColor Blue

# Check if dataset exists
if (Test-Path "data/transactions.csv") {
    Write-Host "✓ Dataset exists (data/transactions.csv)" -ForegroundColor Green
    $lineCount = (Get-Content "data/transactions.csv" | Measure-Object -Line).Lines
    Write-Host "  Row count: $lineCount" -ForegroundColor Gray
} else {
    Write-Host "⚠ Generating dataset..." -ForegroundColor Yellow
    python -m mvp.src.generate_synthetic_dataset --rows 20000 --out data/transactions.csv
    Write-Host "✓ Dataset generated" -ForegroundColor Green
}

# Check index status
Write-Host ""
Write-Host "Checking index status..." -ForegroundColor Blue
$indexMeta = Invoke-RestMethod -Uri "$API_URL/index_meta" -Method GET
$indexMeta | ConvertTo-Json -Depth 10
Write-Host ""

# Step 3: Evaluation Demo
Write-Host "Step 3: Evaluation Demo" -ForegroundColor Blue
Write-Host "----------------------------------------" -ForegroundColor Blue

if (Test-Path "mvp/evaluation/overall_metrics_test.json") {
    Write-Host "✓ Evaluation results found" -ForegroundColor Green
    Write-Host ""
    Write-Host "Overall Metrics:" -ForegroundColor Blue
    $metrics = Get-Content "mvp/evaluation/overall_metrics_test.json" | ConvertFrom-Json
    $metrics | ConvertTo-Json -Depth 10
    Write-Host ""
    Write-Host "Macro F1 Score: $($metrics.macro_f1)" -ForegroundColor Green
} else {
    Write-Host "⚠ Running evaluation..." -ForegroundColor Yellow
    python -m mvp.evaluation.evaluate_comprehensive
    Write-Host "✓ Evaluation complete" -ForegroundColor Green
}
Write-Host ""

# Step 4: Sample Predictions Demo
Write-Host "Step 4: Sample Predictions with Confidence" -ForegroundColor Blue
Write-Host "----------------------------------------" -ForegroundColor Blue

$transactions = @(
    "Zepto grocery order 1234",
    "Swiggy food delivery order 5678",
    "Amazon Prime Video subscription",
    "Uber ride to airport",
    "Netflix monthly payment"
)

foreach ($tx in $transactions) {
    Write-Host ""
    Write-Host "Testing: '$tx'" -ForegroundColor Cyan
    Write-Host "---" -ForegroundColor Gray
    
    $formData = @{
        transaction = $tx
    }
    
    try {
        $response = Invoke-RestMethod -Uri "$API_URL/predict" -Method POST -Form $formData
        Write-Host "Category: $($response.predicted_category_name)" -ForegroundColor Green
        Write-Host "Confidence: $([math]::Round($response.confidence * 100, 2))%" -ForegroundColor Green
        Write-Host "Low Confidence: $($response.is_low_confidence)" -ForegroundColor $(if ($response.is_low_confidence) { "Yellow" } else { "Green" })
        if ($response.keyword_matches) {
            Write-Host "Keywords: $($response.keyword_matches -join ', ')" -ForegroundColor Gray
        }
    } catch {
        Write-Host "Error: $_" -ForegroundColor Red
    }
}
Write-Host ""

# Step 5: Taxonomy Modification Demo
Write-Host "Step 5: Taxonomy Modification Demo" -ForegroundColor Blue
Write-Host "----------------------------------------" -ForegroundColor Blue

# Get current taxonomy
Write-Host "Current taxonomy:" -ForegroundColor Blue
$currentTaxonomy = Invoke-RestMethod -Uri "$API_URL/taxonomy" -Method GET
$currentTaxonomy | ConvertTo-Json -Depth 10 | Select-Object -First 30
Write-Host ""

# Create modified taxonomy
Write-Host "Creating modified taxonomy (adding new alias)..." -ForegroundColor Yellow
$modifiedTaxonomy = @{
    model = "all-MiniLM-L6-v2"
    low_confidence_threshold = 0.5
    categories = @(
        @{
            id = "GROCERIES"
            name = "Groceries"
            aliases = @("zepto", "blinkit", "bigbasket", "grocery", "supermarket", "instamart", "demo_alias")
        },
        @{
            id = "RESTAURANTS"
            name = "Restaurants & Cafes"
            aliases = @("swiggy", "zomato", "restaurant", "cafe", "food delivery")
        },
        @{
            id = "SHOPPING"
            name = "Shopping"
            aliases = @("amazon", "flipkart", "myntra", "shopping", "ecommerce")
        }
    )
}

$tempFile = [System.IO.Path]::GetTempFileName() + ".json"
$modifiedTaxonomy | ConvertTo-Json -Depth 10 | Out-File -FilePath $tempFile -Encoding UTF8

Write-Host "⚠ Uploading modified taxonomy..." -ForegroundColor Yellow
$fileBytes = [System.IO.File]::ReadAllBytes($tempFile)
$boundary = [System.Guid]::NewGuid().ToString()
$fileField = @"
--$boundary
Content-Disposition: form-data; name="file"; filename="taxonomy.json"
Content-Type: application/json

$([System.Text.Encoding]::UTF8.GetString($fileBytes))
--$boundary--
"@

try {
    $uploadResponse = Invoke-RestMethod -Uri "$API_URL/upload_taxonomy" -Method POST `
        -ContentType "multipart/form-data; boundary=$boundary" `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($fileField))
    $uploadResponse | ConvertTo-Json
} catch {
    Write-Host "Note: Upload may require using UI or curl command" -ForegroundColor Yellow
    Write-Host "  curl -X POST $API_URL/upload_taxonomy -F `"file=@$tempFile`"" -ForegroundColor Gray
}

Write-Host ""
Write-Host "⚠ Rebuilding index with new taxonomy..." -ForegroundColor Yellow
$rebuildResponse = Invoke-RestMethod -Uri "$API_URL/rebuild_index" -Method POST
$rebuildResponse | ConvertTo-Json

Write-Host ""
Write-Host "✓ Taxonomy updated and index rebuilt" -ForegroundColor Green

# Cleanup
Remove-Item $tempFile -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Demo Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Blue
Write-Host "1. Open UI: http://localhost:8000/ui/index.html" -ForegroundColor Gray
Write-Host "2. Try more predictions in the UI" -ForegroundColor Gray
Write-Host "3. View evaluation results: mvp/evaluation/" -ForegroundColor Gray
Write-Host "4. Check cost analysis: $API_URL/cost_analysis?scenario=large" -ForegroundColor Gray

