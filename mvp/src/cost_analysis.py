"""
Cost Savings Analysis Module

Calculates ROI and cost savings compared to third-party transaction categorization APIs.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class CostComparison:
    """Cost comparison between in-house solution and third-party APIs."""
    
    # Third-party API costs (per transaction)
    api_cost_per_transaction: float = 0.001  # $0.001 = $1 per 1000 transactions
    
    # In-house solution costs
    infrastructure_cost_per_month: float = 50.0  # Server/hosting costs
    development_cost_one_time: float = 5000.0  # Initial development (amortized)
    maintenance_cost_per_month: float = 200.0  # Ongoing maintenance
    
    # Usage metrics
    transactions_per_month: int = 100000
    
    def calculate_monthly_api_cost(self) -> float:
        """Calculate monthly cost using third-party API."""
        return self.transactions_per_month * self.api_cost_per_transaction
    
    def calculate_monthly_inhouse_cost(self) -> float:
        """Calculate monthly cost for in-house solution."""
        # Amortize development cost over 12 months
        amortized_dev_cost = self.development_cost_one_time / 12
        return (
            self.infrastructure_cost_per_month +
            self.maintenance_cost_per_month +
            amortized_dev_cost
        )
    
    def calculate_monthly_savings(self) -> float:
        """Calculate monthly cost savings."""
        api_cost = self.calculate_monthly_api_cost()
        inhouse_cost = self.calculate_monthly_inhouse_cost()
        return api_cost - inhouse_cost
    
    def calculate_annual_savings(self) -> float:
        """Calculate annual cost savings."""
        return self.calculate_monthly_savings() * 12
    
    def calculate_roi(self, months: int = 12) -> Dict[str, Any]:
        """
        Calculate Return on Investment.
        
        Args:
            months: Number of months to calculate ROI for
        
        Returns:
            Dictionary with ROI metrics
        """
        monthly_api_cost = self.calculate_monthly_api_cost()
        monthly_inhouse_cost = self.calculate_monthly_inhouse_cost()
        monthly_savings = monthly_api_cost - monthly_inhouse_cost
        
        total_api_cost = monthly_api_cost * months
        total_inhouse_cost = (
            self.development_cost_one_time +
            (self.infrastructure_cost_per_month + self.maintenance_cost_per_month) * months
        )
        total_savings = total_api_cost - total_inhouse_cost
        
        # Calculate ROI percentage (handle division by zero and ensure finite values)
        if self.development_cost_one_time > 0:
            roi_percentage = ((total_savings - self.development_cost_one_time) / self.development_cost_one_time) * 100
            # Ensure ROI is finite
            if not isinstance(roi_percentage, (int, float)) or not (-1e10 < roi_percentage < 1e10):
                roi_percentage = 0
        else:
            roi_percentage = 0
        
        # Calculate payback period (avoid inf for JSON compatibility)
        if monthly_savings > 0:
            payback_period_months = self.development_cost_one_time / monthly_savings
        else:
            payback_period_months = None  # No payback if monthly savings are negative/zero
        
        # Ensure ROI percentage is finite (handle edge cases)
        if not isinstance(roi_percentage, (int, float)) or not (-1e10 < roi_percentage < 1e10):
            roi_percentage = None
        
        return {
            "period_months": months,
            "total_api_cost": round(total_api_cost, 2),
            "total_inhouse_cost": round(total_inhouse_cost, 2),
            "total_savings": round(total_savings, 2),
            "roi_percentage": round(roi_percentage, 2) if roi_percentage is not None else None,
            "payback_period_months": round(payback_period_months, 2) if payback_period_months is not None else None,
            "monthly_api_cost": round(monthly_api_cost, 2),
            "monthly_inhouse_cost": round(monthly_inhouse_cost, 2),
            "monthly_savings": round(monthly_savings, 2),
            "break_even_month": int(payback_period_months) if payback_period_months is not None and payback_period_months < 1e6 else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        monthly_api = self.calculate_monthly_api_cost()
        monthly_inhouse = self.calculate_monthly_inhouse_cost()
        monthly_savings = self.calculate_monthly_savings()
        
        return {
            "transactions_per_month": self.transactions_per_month,
            "api_cost_per_transaction": self.api_cost_per_transaction,
            "monthly_api_cost": round(monthly_api, 2),
            "monthly_inhouse_cost": round(monthly_inhouse, 2),
            "monthly_savings": round(monthly_savings, 2),
            "annual_savings": round(self.calculate_annual_savings(), 2),
            "savings_percentage": round((monthly_savings / monthly_api * 100) if monthly_api > 0 else 0, 2),
            "roi_12_months": self.calculate_roi(months=12),
            "roi_24_months": self.calculate_roi(months=24),
            "cost_breakdown": {
                "infrastructure_per_month": self.infrastructure_cost_per_month,
                "maintenance_per_month": self.maintenance_cost_per_month,
                "development_one_time": self.development_cost_one_time,
                "development_amortized_12mo": round(self.development_cost_one_time / 12, 2)
            }
        }


# Default cost comparison scenarios
# Note: Costs are realistic estimates showing actual savings for in-house solution
DEFAULT_SCENARIOS = {
    "small": CostComparison(
        transactions_per_month=10000,
        api_cost_per_transaction=0.002,  # $2 per 1000 transactions (typical API pricing)
        infrastructure_cost_per_month=15.0,  # Light server/hosting
        development_cost_one_time=1500.0,  # One-time development (this MVP)
        maintenance_cost_per_month=50.0  # Minimal maintenance
    ),
    "medium": CostComparison(
        transactions_per_month=100000,
        api_cost_per_transaction=0.002,  # $2 per 1000 = $200/month for 100k
        infrastructure_cost_per_month=30.0,  # Moderate server (can handle 100k+ transactions)
        development_cost_one_time=3000.0,  # One-time development cost
        maintenance_cost_per_month=100.0  # Regular maintenance
    ),
    "large": CostComparison(
        transactions_per_month=1000000,
        api_cost_per_transaction=0.0015,  # Volume discount, but still $1,500/month
        infrastructure_cost_per_month=100.0,  # Larger server for scale
        development_cost_one_time=5000.0,  # One-time development
        maintenance_cost_per_month=200.0  # More maintenance at scale
    )
}


def get_cost_analysis(scenario: str = "medium") -> Dict[str, Any]:
    """
    Get cost analysis for a given scenario.
    
    Args:
        scenario: One of "small", "medium", "large"
    
    Returns:
        Dictionary with cost analysis
    """
    if scenario not in DEFAULT_SCENARIOS:
        scenario = "medium"
    
    comparison = DEFAULT_SCENARIOS[scenario]
    return comparison.to_dict()

