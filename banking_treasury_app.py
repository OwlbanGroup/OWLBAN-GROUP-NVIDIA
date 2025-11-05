#!/usr/bin/env python3
"""
OWLBAN GROUP Banking Treasury Application
JPMorgan Integration - Treasury Management Module
"""

import logging
import json
from typing import Dict, List, Any
from jpmorgan_api_integration import JPMorganAPIIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BankingTreasuryApp")

class BankingTreasuryApp:
    """Banking Treasury Application using JPMorgan Integration"""

    def __init__(self, environment: str = "sandbox"):
        self.jpmorgan = JPMorganAPIIntegration(environment)
        self.logger = logger
        self.accounts = {}  # Cache for account information

    def authenticate(self) -> bool:
        """Authenticate with JPMorgan"""
        return self.jpmorgan.authenticate()

    def get_account_balance(self, account_id: str) -> Dict[str, Any]:
        """Get account balance"""
        self.logger.info(f"Retrieving balance for account: {account_id}")

        balance = self.jpmorgan.get_account_balance(account_id)
        if balance.get("status") != "failed":
            self.accounts[account_id] = balance

        return balance

    def transfer_funds(self, from_account: str, to_account: str, amount: float,
                      currency: str = "USD", description: str = "") -> Dict[str, Any]:
        """Transfer funds between accounts"""
        self.logger.info(f"Transferring ${amount} {currency} from {from_account} to {to_account}")

        result = self.jpmorgan.transfer_funds(from_account, to_account, amount, currency)

        if result.get("status") == "completed":
            # Update cached balances
            if from_account in self.accounts:
                self.accounts[from_account]["availableBalance"]["amount"] -= amount
            if to_account in self.accounts:
                self.accounts[to_account]["availableBalance"]["amount"] += amount

        return result

    def optimize_cash_position(self, accounts: List[str]) -> Dict[str, Any]:
        """Optimize cash position across accounts"""
        self.logger.info(f"Optimizing cash position for {len(accounts)} accounts")

        total_balance = 0
        account_balances = {}

        # Get balances for all accounts
        for account_id in accounts:
            balance = self.get_account_balance(account_id)
            if balance.get("status") != "failed":
                available = balance.get("availableBalance", {}).get("amount", 0)
                account_balances[account_id] = available
                total_balance += available

        # Simple optimization: balance accounts evenly
        target_balance = total_balance / len(accounts)
        optimization_actions = []

        for account_id, current_balance in account_balances.items():
            if current_balance < target_balance * 0.9:  # Need more cash
                needed = target_balance - current_balance
                # Find account with surplus
                for surplus_account, surplus_balance in account_balances.items():
                    if surplus_balance > target_balance * 1.1 and surplus_account != account_id:
                        transfer_amount = min(needed, surplus_balance - target_balance)
                        if transfer_amount > 0:
                            optimization_actions.append({
                                "action": "transfer",
                                "from_account": surplus_account,
                                "to_account": account_id,
                                "amount": transfer_amount,
                                "reason": "Cash optimization"
                            })
                        break

        # Execute optimization actions
        results = []
        for action in optimization_actions:
            result = self.transfer_funds(
                action["from_account"],
                action["to_account"],
                action["amount"],
                description=action["reason"]
            )
            results.append(result)

        return {
            "total_balance": total_balance,
            "target_balance_per_account": target_balance,
            "account_balances": account_balances,
            "optimization_actions": len(optimization_actions),
            "executed_transfers": results,
            "status": "completed"
        }

    def forecast_cash_flow(self, account_id: str, days: int = 30) -> Dict[str, Any]:
        """Forecast cash flow for an account"""
        self.logger.info(f"Forecasting cash flow for {account_id} over {days} days")

        # Get current balance
        balance = self.get_account_balance(account_id)
        current_balance = balance.get("availableBalance", {}).get("amount", 0)

        # Simulate cash flow forecast (in real implementation, this would use historical data and AI)
        forecast = []
        balance_projection = current_balance

        for day in range(1, days + 1):
            # Simulate daily inflows and outflows
            inflow = 5000 + (day * 100)  # Increasing inflows
            outflow = 3000 + (day * 50)  # Increasing outflows
            net_flow = inflow - outflow

            balance_projection += net_flow

            forecast.append({
                "day": day,
                "inflow": inflow,
                "outflow": outflow,
                "net_flow": net_flow,
                "projected_balance": balance_projection
            })

        return {
            "account_id": account_id,
            "current_balance": current_balance,
            "forecast_period_days": days,
            "forecast": forecast,
            "final_projected_balance": balance_projection,
            "status": "completed"
        }

    def generate_treasury_report(self, accounts: List[str]) -> str:
        """Generate comprehensive treasury report"""
        self.logger.info(f"Generating treasury report for {len(accounts)} accounts")

        total_balance = 0
        account_details = []

        for account_id in accounts:
            balance = self.get_account_balance(account_id)
            if balance.get("status") != "failed":
                available = balance.get("availableBalance", {}).get("amount", 0)
                current = balance.get("currentBalance", {}).get("amount", 0)
                total_balance += available

                account_details.append({
                    "account_id": account_id,
                    "available_balance": available,
                    "current_balance": current,
                    "status": balance.get("status", "unknown")
                })

        # Cash flow forecast for primary account
        if accounts:
            forecast = self.forecast_cash_flow(accounts[0], 7)  # 7-day forecast

        report = f"""
# Banking Treasury Application Report
## OWLBAN GROUP - JPMorgan Integration

**Report Date:** 2024-01-15
**Environment:** {self.jpmorgan.environment.upper()}

---

## Treasury Overview

- **Total Accounts:** {len(accounts)}
- **Total Available Balance:** ${total_balance:,.2f}
- **Average Balance per Account:** ${total_balance/len(accounts):,.2f}

## Account Details

{'| Account ID | Available Balance | Current Balance | Status |'}
{'|------------|------------------|-----------------|--------|'}

"""

        for account in account_details:
            report += f"| {account['account_id']} | ${account['available_balance']:,.2f} | ${account['current_balance']:,.2f} | {account['status']} |\n"

        report += f"""

## Cash Flow Forecast (7 Days)

| Day | Inflow | Outflow | Net Flow | Projected Balance |
|-----|--------|---------|----------|-------------------|

"""

        if accounts and 'forecast' in locals():
            for day_data in forecast["forecast"][:7]:  # Show first 7 days
                report += f"| {day_data['day']} | ${day_data['inflow']:,.2f} | ${day_data['outflow']:,.2f} | ${day_data['net_flow']:,.2f} | ${day_data['projected_balance']:,.2f} |\n"

        report += """

## Treasury Metrics

- **Liquidity Ratio:** 1.8 (Healthy > 1.5)
- **Cash Conversion Cycle:** 12 days
- **Working Capital:** $2.3M
- **Debt-to-Equity Ratio:** 0.3

## Recommendations

- **Cash Optimization:** Consider balancing funds across accounts
- **Liquidity Management:** Maintain minimum 15% cash reserves
- **Investment Opportunities:** Excess cash available for short-term investments

---

**OWLBAN GROUP Banking Treasury Application**
**Powered by JPMorgan Integration**

"""

        return report

def main():
    """Demonstrate Banking Treasury Application"""
    print("OWLBAN GROUP - Banking Treasury Application")
    print("=" * 50)

    # Initialize treasury app
    treasury_app = BankingTreasuryApp(environment="sandbox")

    # Authenticate
    if not treasury_app.authenticate():
        print("❌ Authentication failed")
        return

    print("✅ Authentication successful")

    # Get account balances
    accounts = ["OWL001234", "TREASURY999", "OPERATIONS888"]
    print("\n🏦 Retrieving Account Balances...")

    for account_id in accounts:
        balance = treasury_app.get_account_balance(account_id)
        available = balance.get("availableBalance", {}).get("amount", 0)
        print(f"Account {account_id}: ${available:,.2f}")

    # Transfer funds
    print("\n🔄 Executing Fund Transfer...")
    transfer = treasury_app.transfer_funds(
        "OWL001234",
        "TREASURY999",
        25000.00,
        description="Treasury optimization"
    )
    print(f"Transfer Result: {transfer.get('status')}")

    # Optimize cash position
    print("\n📊 Optimizing Cash Position...")
    optimization = treasury_app.optimize_cash_position(accounts)
    print(f"Optimization completed: {optimization['optimization_actions']} actions executed")

    # Generate treasury report
    report = treasury_app.generate_treasury_report(accounts)

    with open('banking_treasury_report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n📋 Treasury report saved to 'banking_treasury_report.md'")
    print("🎉 Banking Treasury Application Demo Complete!")

if __name__ == "__main__":
    main()
