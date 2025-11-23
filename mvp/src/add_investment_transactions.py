"""
Add Indian Investment Application Transactions to Dataset

This script generates investment-related transactions for Indian investment apps
and appends them to the existing transactions.csv file.
"""

import csv
import random
import os
from pathlib import Path
from typing import List, Dict

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "transactions.csv"

# Indian Investment Apps and Services
INVESTMENT_MERCHANTS = [
    "Zerodha",
    "Upstox",
    "Groww",
    "Angel One",
    "ICICI Direct",
    "HDFC Securities",
    "Sharekhan",
    "Motilal Oswal",
    "5paisa",
    "Paytm Money",
    "Kotak Securities",
    "Axis Direct",
    "IIFL Securities",
    "Edelweiss",
    "Geojit Financial",
]

# Investment transaction types
INVESTMENT_TYPES = [
    "SIP Investment",
    "Mutual Fund Purchase",
    "Equity Purchase",
    "Stock Trading",
    "Demat Charges",
    "Brokerage Fee",
    "AMC Charges",
    "Transaction Charges",
    "Investment",
    "Trading",
    "Equity SIP",
    "Debt Fund",
    "Hybrid Fund",
    "ELSS Investment",
    "NFO Subscription",
]

# Investment-specific noise templates
INVESTMENT_TEMPLATES = [
    "{merchant} {type} {amt}",
    "{merchant} {type} #{id}",
    "{merchant} {type} UPI {id}",
    "{merchant} {type} NEFT {id}",
    "{merchant} {type} {amt} INR",
    "{merchant} {type} order {id}",
    "{merchant} {type} - {amt}",
    "{merchant} {type} payment {id}",
    "{merchant} {type} transaction {id}",
    "{merchant} {type} debit {amt}",
    "{merchant} SIP {amt}",
    "{merchant} Mutual Fund {amt}",
    "{merchant} Equity {amt}",
    "{merchant} Demat {amt}",
    "{merchant} Brokerage {amt}",
    "{merchant} {type} Mumbai",
    "{merchant} {type} Bangalore",
    "{merchant} {type} Delhi",
]

# Investment-specific misspellings
INVESTMENT_MISPELLINGS = {
    "Zerodha": ["Zerodha App", "Zerodha Kite", "Zerodha Coin", "Zerodha Trading"],
    "Upstox": ["Upstox Pro", "Upstox App", "Upstox Trading"],
    "Groww": ["Groww App", "Groww Investment", "Groww Mutual Fund"],
    "Angel One": ["Angel One Trading", "Angel Broking", "Angel One App"],
    "ICICI Direct": ["ICICI Direct Trading", "ICICI Securities", "ICICI Direct App"],
    "HDFC Securities": ["HDFC Sec", "HDFC Securities Trading", "HDFC Securities App"],
}

def noisy_merchant(name: str) -> str:
    """Add noise/variations to merchant names."""
    if name in INVESTMENT_MISPELLINGS and random.random() < 0.4:
        return random.choice(INVESTMENT_MISPELLINGS[name])
    return name

def generate_investment_transaction() -> Dict[str, str]:
    """Generate a single investment transaction."""
    merchant = noisy_merchant(random.choice(INVESTMENT_MERCHANTS))
    inv_type = random.choice(INVESTMENT_TYPES)
    template = random.choice(INVESTMENT_TEMPLATES)
    
    # Generate realistic amounts for investments
    if "SIP" in inv_type or "sip" in template.lower():
        amt = f"{random.randint(500, 10000)}.{random.randint(0, 99):02d}"
    elif "Brokerage" in inv_type or "brokerage" in template.lower():
        amt = f"{random.randint(10, 500)}.{random.randint(0, 99):02d}"
    elif "Demat" in inv_type or "demat" in template.lower():
        amt = f"{random.randint(0, 1000)}.{random.randint(0, 99):02d}"
    else:
        amt = f"{random.randint(1000, 50000)}.{random.randint(0, 99):02d}"
    
    tid = random.randint(1000, 99999)
    
    description = template.format(
        merchant=merchant,
        type=inv_type,
        amt=amt,
        id=tid
    )
    
    return {"description": description, "category": "INVESTMENT"}

def add_investment_transactions(num_transactions: int = 2000, append: bool = True):
    """
    Generate investment transactions and add to CSV.
    
    Args:
        num_transactions: Number of investment transactions to generate
        append: If True, append to existing CSV. If False, create new file.
    """
    # Generate transactions
    print(f"Generating {num_transactions} investment transactions...")
    transactions = []
    for _ in range(num_transactions):
        transactions.append(generate_investment_transaction())
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and has header
    file_exists = CSV_PATH.exists()
    has_header = False
    
    if file_exists:
        # Check if file has content
        with open(CSV_PATH, 'r', encoding='utf8', newline='') as f:
            reader = csv.reader(f)
            try:
                first_row = next(reader)
                has_header = first_row == ['description', 'category']
            except StopIteration:
                file_exists = False
    
    # Write transactions
    mode = 'a' if append and file_exists else 'w'
    with open(CSV_PATH, mode, encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['description', 'category'])
        
        # Write header if creating new file or appending to empty file
        if not (append and has_header):
            writer.writeheader()
        
        writer.writerows(transactions)
    
    action = "Appended" if append and file_exists else "Created"
    print(f"[OK] {action} {num_transactions} investment transactions to {CSV_PATH}")
    
    # Count total rows
    if file_exists:
        with open(CSV_PATH, 'r', encoding='utf8') as f:
            total_rows = sum(1 for line in f) - 1  # Subtract header
        print(f"[OK] Total transactions in file: {total_rows}")
    
    return CSV_PATH

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add investment transactions to dataset")
    parser.add_argument("--count", type=int, default=2000, help="Number of investment transactions to add (default: 2000)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing file instead of appending")
    
    args = parser.parse_args()
    
    add_investment_transactions(
        num_transactions=args.count,
        append=not args.overwrite
    )

