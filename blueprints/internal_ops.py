"""
Internal Operations Blueprint
Orchestrates internal team payroll, internal personal banking, and company bill payments.
"""

from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, g

try:
    from src.auth import token_auth_required
except ImportError:
    def token_auth_required(f):
        return f

try:
    from src.rate_limiting import conditional_limit
except ImportError:
    def conditional_limit(_limit):
        def decorator(f):
            return f
        return decorator

try:
    from src.logger import telemetry_logger
except ImportError:
    class _FallbackLogger:
        def log_info(self, msg, context=None):
            print(f"INFO: {msg} | {context}")
        def log_error(self, msg, context=None):
            print(f"ERROR: {msg} | {context}")
    telemetry_logger = _FallbackLogger()

try:
    from src.payroll_service import payroll_service
except ImportError:
    payroll_service = None


internal_ops_bp = Blueprint("internal_ops", __name__)


@internal_ops_bp.route("/execute", methods=["POST"])
@token_auth_required
@conditional_limit("10 per minute")
def execute_internal_operations():
    """
    Execute internal operations in one request:
      1) Internal team payroll
      2) Internal personal banking (PFM-like summary actions)
      3) Company bill payments
    """
    try:
        payload = request.get_json() or {}
        user_id = g.get("user_id", payload.get("user_id", "internal_ops_user"))

        payroll_req = payload.get("payroll", {})
        personal_banking_req = payload.get("personal_banking", {})
        company_bills_req = payload.get("company_bills", {})

        if not payroll_req and not personal_banking_req and not company_bills_req:
            return jsonify({
                "status": "error",
                "message": "At least one section is required: payroll, personal_banking, company_bills"
            }), 400

        results = {
            "payroll": None,
            "personal_banking": None,
            "company_bills": None
        }

        # 1) Payroll orchestration
        if payroll_req:
            if payroll_service is None:
                results["payroll"] = {
                    "status": "error",
                    "message": "Payroll service unavailable"
                }
            else:
                run_input = {
                    "pay_period_start": payroll_req.get("pay_period_start"),
                    "pay_period_end": payroll_req.get("pay_period_end"),
                    "payment_date": payroll_req.get("payment_date")
                }
                if not all(run_input.values()):
                    results["payroll"] = {
                        "status": "error",
                        "message": "payroll requires pay_period_start, pay_period_end, payment_date"
                    }
                else:
                    created = payroll_service.create_payroll_run(user_id, run_input)
                    if created.get("status") == "success":
                        run_id = created["run"]["run_id"]
                        processed = payroll_service.process_payroll_run(run_id)
                        results["payroll"] = processed
                    else:
                        results["payroll"] = created

        # 2) Internal personal banking orchestration (mock summary)
        if personal_banking_req:
            results["personal_banking"] = {
                "status": "success",
                "actions": {
                    "account_review": bool(personal_banking_req.get("account_review", True)),
                    "budget_check": bool(personal_banking_req.get("budget_check", True)),
                    "bill_reminders_check": bool(personal_banking_req.get("bill_reminders_check", True))
                },
                "message": "Internal personal banking workflow executed"
            }

        # 3) Company bill pay orchestration (mock batch)
        if company_bills_req:
            bills = company_bills_req.get("bills", [])
            if not isinstance(bills, list):
                results["company_bills"] = {
                    "status": "error",
                    "message": "company_bills.bills must be a list"
                }
            else:
                processed_bills = []
                total_amount = 0.0
                for b in bills:
                    amount = float(b.get("amount", 0) or 0)
                    total_amount += amount
                    processed_bills.append({
                        "vendor": b.get("vendor", "unknown"),
                        "amount": amount,
                        "currency": b.get("currency", "USD"),
                        "status": "scheduled"
                    })

                results["company_bills"] = {
                    "status": "success",
                    "count": len(processed_bills),
                    "total_amount": round(total_amount, 2),
                    "payments": processed_bills,
                    "message": "Company bill payments scheduled"
                }

        telemetry_logger.log_info(
            "Internal operations executed",
            {"user_id": user_id, "context": "internal_ops"}
        )

        return jsonify({
            "status": "success",
            "workflow": "internal_operations",
            "results": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(str(e), {"context": "execute_internal_operations"})
        return jsonify({"status": "error", "message": "Internal server error"}), 500


@internal_ops_bp.route("/health", methods=["GET"])
def internal_ops_health():
    return jsonify({
        "status": "healthy",
        "service": "internal_ops",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


__all__ = ["internal_ops_bp"]
