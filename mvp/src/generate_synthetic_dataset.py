"""
Generate a synthetic, India-heavy transactions.csv for the Tx Categorization MVP.

Creates: mvp/data/transactions.csv
Columns: description, category
- Includes noisy forms, Indian merchants, UPI/INR hints
- Default: 20,000 rows across 13 categories (configurable via --rows)
"""

import argparse
import csv
import os
import random
from typing import List, Dict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_CSV = os.path.join(DATA_DIR, "transactions.csv")

os.makedirs(DATA_DIR, exist_ok=True)

CATEGORIES: Dict[str, List[str]] = {
    "GROCERIES": [
        "Zepto",
        "Blinkit",
        "BigBasket",
        "More Supermarket",
        "Reliance Fresh",
        "Nature's Basket",
        "Grocery Market",
    ],
    "RESTAURANTS": [
        "Swiggy",
        "Zomato",
        "Dominos",
        "Haldirams",
        "Barbeque Nation",
        "Indian Coffee House",
        "Cafe Coffee Day",
    ],
    "TRANSPORT": [
        "Uber",
        "Ola",
        "Rapido",
        "Airport Taxi",
        "RedBus",
        "IRCTC",
    ],
    "FUEL": [
        "Indian Oil",
        "Bharat Petroleum",
        "HPCL Fuel",
        "Reliance Fuel",
    ],
    "SHOPPING": [
        "Amazon India",
        "Flipkart",
        "Myntra",
        "Ajio",
        "Nykaa",
        "Croma",
        "Tanishq",
    ],
    "UTILITIES": [
        "BESCOM",
        "MSEB Electricity",
        "BWSSB Water",
        "Mahanagar Gas",
        "Airtel Fiber",
        "JioFiber",
    ],
    "RENT": [
        "Rent Payment",
        "NoBroker Rent",
        "NestAway",
        "Landlord",
        "Apartment Dues",
    ],
    "INCOME": [
        "Salary Credit",
        "HDFC Payroll",
        "ICICI Payroll",
        "Employer Inc",
        "Paycheck",
    ],
    "BANKING": [
        "HDFC Bank",
        "ICICI Bank",
        "SBI Bank",
        "Axis Bank",
        "Kotak Bank",
    ],
    "HEALTH": [
        "Apollo Clinic",
        "Fortis Hospital",
        "1mg Pharmacy",
        "Pharmeasy",
        "Doctor Visit",
    ],
    "ENTERTAINMENT": [
        "Netflix",
        "Disney+ Hotstar",
        "Sony LIV",
        "Amazon Prime Video",
        "Prime Music",
        "BookMyShow",
        "PVR Cinemas",
        "Gaana",
        "Spotify",
    ],
    "SUBSCRIPTIONS": [
        "YouTube Premium",
        "Headspace",
        "Cult.fit",
        "AWS Subscription",
        "Amazon Web Services",
        "Canva Pro",
        "Dropbox plan",
    ],
    "EDUCATION": [
        "Byju's",
        "Unacademy",
        "UpGrad",
        "Coursera",
        "Udemy",
        "Great Learning",
        "School Payment",
        "College Fee",
    ],
    "TRAVEL": [
        "IndiGo",
        "Air India",
        "GoFirst",
        "Vistara",
        "MakeMyTrip",
        "Yatra.com",
        "OYO Rooms",
        "Hotel Booking",
    ],
    "INVESTMENT": [
        "Zerodha",
        "Upstox",
        "Groww",
        "Angel One",
        "ICICI Direct",
        "HDFC Securities",
        "Sharekhan",
        "Motilal Oswal",
        "SIP Investment",
        "Mutual Fund",
    ],
}

NOISE_TEMPLATES = [
    "{merchant} {amt}",
    "{merchant} #{id}",
    "{merchant} order {id}",
    "{merchant} POS {id}",
    "{merchant} online",
    "{merchant} +tax {amt}",
    "{merchant} -promo",
    "{merchant} IN",
    "{merchant} {amt} INR",
    "{merchant} UPI {id}",
    "{merchant} GPay txn {id}",
    "{merchant} NEFT {id}",
    "{merchant} ACH {id}",
    "{merchant} Bangalore",
    "{merchant} Mumbai",
]

MISPELLINGS = {
    "Swiggy": ["Swggy", "Sviggy"],
    "Zomato": ["Zommato", "Zmt"],
    "Zepto": ["Zepto App", "Zept0"],
    "Indian Oil": ["India Oil", "IndianOil"],
    "HDFC Bank": ["HDFC", "Hdfc Bank"],
    "Amazon India": ["Amazon.in", "Amzn India"],
    "Flipkart": ["Flip cart", "Flpkart"],
    "Myntra": ["Myntr", "Myntra.com"],
    "Uber": ["Ubr", "Uber India"],
    "Ola": ["Ola Cabs", "OLA"],
}

def noisy_merchant(name: str) -> str:
    if name in MISPELLINGS and random.random() < 0.3:
        return random.choice(MISPELLINGS[name])
    return name

def gen_rows(n: int = 2000):
    rows = []
    cats = list(CATEGORIES.items())
    per_cat = max(10, n // len(cats))
    for cat_id, merchants in cats:
        for _ in range(per_cat):
            rows.append(_row_for(cat_id, merchants))
    while len(rows) < n:
        cat_id, merchants = random.choice(cats)
        rows.append(_row_for(cat_id, merchants))
    random.shuffle(rows)
    return rows

def _row_for(cat_id: str, merchants: List[str]) -> Dict[str, str]:
    merchant = noisy_merchant(random.choice(merchants))
    tmpl = random.choice(NOISE_TEMPLATES)
    amt = f"{random.randint(50, 9999)}.{random.randint(0,99):02d}"
    tid = random.randint(100, 9999)
    desc = tmpl.format(merchant=merchant, amt=amt, id=tid)
    return {"description": desc, "category": cat_id}

def save_csv(rows, path=OUT_CSV):
    with open(path, "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["description", "category"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction dataset")
    parser.add_argument("--rows", type=int, default=20000, help="How many rows to generate (default: 20000)")
    parser.add_argument("--out", type=str, default=OUT_CSV, help="Destination CSV path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    rows = gen_rows(args.rows)
    save_csv(rows, args.out)
