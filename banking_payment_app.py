"""OWLBAN banking payment application backed by the JPMorgan simulation."""

from __future__ import annotations

from typing import Any, Dict

from jpmorgan_api_integration import (
    JPMorganAPIIntegration,
    PaymentRequest,
)


class BankingPaymentApp:
    def __init__(self, environment: str = "sandbox") -> None:
        self.jpmorgan = JPMorganAPIIntegration(environment)

    def validate_payment(self, payment: Dict[str, Any]) -> bool:
        amount = payment.get("amount")
        return isinstance(amount, (int, float)) and amount > 0

    def process_payment(self, payment: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_payment(payment):
            return {"status": "failed", "error": "Invalid payment"}
        request = PaymentRequest(
            amount=float(payment["amount"]),
            recipient_account=str(payment.get("recipient", payment.get("recipient_account", ""))),
            sender_account=str(payment.get("sender", payment.get("sender_account", ""))),
            currency=str(payment.get("currency", "USD")),
        )
        result = self.jpmorgan.process_payment(request)
        return {
            "status": result.get("status", "failed"),
            "transactionId": result.get("transactionId"),
            "exchange_rate": result.get("exchange_rate", 1.0),
        }

    def initiate_transfer(self, transfer: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_payment(transfer):
            return {"status": "failed", "currency": transfer.get("currency", "USD")}
        return {
            "status": "initiated",
            "currency": transfer["currency"],
            "exchange_rate": 1.0,
        }

    def process_international_transfer(self, transfer: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_payment(transfer):
            return {"status": "failed"}
        return {
            "status": "processed",
            "swift_reference": "SWIFT123",
            "currency": transfer.get("currency", "USD"),
        }
