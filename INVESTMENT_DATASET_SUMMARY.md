# Investment Dataset Addition Summary

## Overview

Successfully added **2,000 investment-related transactions** to the dataset for Indian investment applications.

## Statistics

- **Total Transactions Added**: 2,000
- **Total Transactions in Dataset**: 24,000 (20,000 original + 2,000 investment)
- **Category**: INVESTMENT
- **File Location**: `data/transactions.csv`

## Investment Applications Included

The dataset now includes transactions from the following Indian investment platforms:

1. **Zerodha** (including variations: Zerodha App, Zerodha Kite, Zerodha Coin)
2. **Upstox** (including variations: Upstox Pro, Upstox App)
3. **Groww** (including variations: Groww App, Groww Investment)
4. **Angel One** (including variations: Angel Broking, Angel One Trading)
5. **ICICI Direct**
6. **HDFC Securities**
7. **Sharekhan**
8. **Motilal Oswal**
9. **5paisa**
10. **Paytm Money**
11. **Kotak Securities**
12. **Axis Direct**
13. **IIFL Securities**
14. **Edelweiss**
15. **Geojit Financial**

## Transaction Types Included

The dataset includes various investment transaction types:

- **SIP Investments** (Systematic Investment Plans)
- **Mutual Fund Purchases**
- **Equity Purchases**
- **Stock Trading**
- **Demat Account Charges**
- **Brokerage Fees**
- **AMC Charges** (Annual Maintenance Charges)
- **Transaction Charges**
- **ELSS Investments** (Equity Linked Savings Scheme)
- **NFO Subscriptions** (New Fund Offer)
- **Debt Fund Investments**
- **Hybrid Fund Investments**

## Sample Transactions

Here are some examples of the generated transactions:

```
HDFC Securities ELSS Investment Bangalore
Motilal Oswal SIP 4451.99
Sharekhan ELSS Investment order 90135
Paytm Money NFO Subscription Delhi
Edelweiss Equity 912.69
Kotak Securities Stock Trading Bangalore
Groww Mutual Fund Equity 41294.81
5paisa Brokerage 118.52
Zerodha Trading Hybrid Fund Bangalore
Paytm Money Demat 311.76
Groww SIP 2986.75
Zerodha App Debt Fund 2020.92 INR
Angel Broking Investment Mumbai
```

## Transaction Formats

Transactions include realistic variations:
- With amounts: `Groww SIP 2986.75`
- With transaction IDs: `Zerodha Trading order 45363`
- With payment methods: `Paytm Money NEFT 4634`
- With locations: `HDFC Securities ELSS Investment Bangalore`
- With INR suffix: `Edelweiss Equity 912.69 INR`

## Next Steps

1. **Rebuild the Index**:
   ```bash
   curl -X POST http://localhost:8000/rebuild_index
   ```
   Or click "Rebuild Index" in the UI.

2. **Test Predictions**:
   - Try: `"Zerodha SIP 5000"`
   - Try: `"Upstox trading order 12345"`
   - Try: `"Groww mutual fund purchase"`

3. **Verify Categorization**:
   All investment-related transactions should now be categorized as **INVESTMENT** instead of other categories.

## Script Usage

To add more investment transactions in the future:

```bash
# Add 1000 more investment transactions
python -m mvp.src.add_investment_transactions --count 1000

# Overwrite and create new file (use with caution)
python -m mvp.src.add_investment_transactions --count 5000 --overwrite
```

## Verification

After rebuilding the index, test with these queries:
- `"zerodha"` → Should return INVESTMENT
- `"upstox"` → Should return INVESTMENT
- `"groww sip"` → Should return INVESTMENT
- `"angel one trading"` → Should return INVESTMENT
- `"hdfc securities"` → Should return INVESTMENT

---

*Generated: 2025-01-XX*
*Script: `mvp/src/add_investment_transactions.py`*

