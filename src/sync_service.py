"""
Synchronization service for linking payments and revenue transactions
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from src.payments_service import payments_service
from src.revenue_service import revenue_service
from src.ai_service import ai_service
from src.models.revenue import RevenueType
from src.logger import telemetry_logger

class SyncService:
    """Service for synchronizing payments and revenue data"""

    def __init__(self):
        self.logger = telemetry_logger.get_logger()

    def sync_payment_to_revenue(self, payment_id: str, revenue_type: RevenueType = RevenueType.PURCHASE) -> Dict[str, Any]:
        """
        Sync a payment transaction to create corresponding revenue transaction

        Args:
            payment_id: Payment transaction ID
            revenue_type: Type of revenue transaction

        Returns:
            Sync result with transaction details
        """
        try:
            # Get payment details
            payment = payments_service.get_payment(payment_id)
            if not payment:
                return {
                    "status": "error",
                    "error": f"Payment {payment_id} not found",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            # Create revenue transaction
            revenue_transaction = revenue_service.create_transaction(
                user_id=payment.user_id,
                revenue_type=revenue_type,
                amount=payment.amount,
                currency=payment.currency,
                description=f"Payment sync: {payment.description or 'Payment transaction'}",
                payment_method=payment.payment_type,
                external_reference=payment_id,
                additional_metadata={
                    "payment_id": payment_id,
                    "sync_timestamp": datetime.now(timezone.utc).isoformat(),
                    "sync_source": "payment_sync"
                }
            )

            self.logger.info(f"Synced payment {payment_id} to revenue transaction {revenue_transaction.transaction_id}")

            return {
                "status": "success",
                "payment_id": payment_id,
                "revenue_transaction_id": revenue_transaction.transaction_id,
                "amount": payment.amount,
                "revenue_type": revenue_type.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to sync payment {payment_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "payment_id": payment_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_business_intelligence(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive business intelligence for a user

        Args:
            user_id: User identifier
            days: Number of days to analyze

        Returns:
            Business intelligence report
        """
        try:
            from datetime import timedelta

            # Get date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            # Get revenue metrics
            revenue_metrics = revenue_service.get_revenue_metrics(start_date, end_date)

            # Get user payments
            user_payments = payments_service.get_user_payments(user_id, limit=1000)

            # Get payment stats
            payment_stats = payments_service.get_payment_stats()

            # Prepare data for AI analysis
            financial_data = {
                "revenue_metrics": revenue_metrics,
                "payment_stats": payment_stats,
                "user_payments_count": len(user_payments),
                "analysis_period_days": days,
                "total_revenue": revenue_metrics.get("net_revenue", 0),
                "total_payments": payment_stats.get("total_amount", 0)
            }

            # Get AI insights
            ai_analysis = ai_service.analyze_financial_data(
                data=financial_data,
                question="Provide comprehensive business intelligence analysis including trends, recommendations, and growth opportunities."
            )

            # Get AI risk assessment
            risk_assessment = ai_service.assess_transaction_risk(
                transaction_data=financial_data,
                historical_patterns=[p.to_dict() for p in user_payments[-10:]] if user_payments else []
            )

            return {
                "status": "success",
                "user_id": user_id,
                "analysis_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "financial_summary": {
                    "total_revenue": revenue_metrics.get("net_revenue", 0),
                    "total_payments": payment_stats.get("total_amount", 0),
                    "transaction_count": revenue_metrics.get("transaction_count", 0),
                    "average_transaction_value": revenue_metrics.get("average_transaction_value", 0),
                    "completion_rate": payment_stats.get("completion_rate", 0)
                },
                "ai_insights": ai_analysis.get("analysis", "AI analysis unavailable"),
                "risk_assessment": risk_assessment.get("risk_assessment", "Risk assessment unavailable"),
                "recommendations": self._generate_business_recommendations(financial_data),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to get business intelligence for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def forecast_revenue(self, user_id: str, forecast_days: int = 30) -> Dict[str, Any]:
        """
        Forecast future revenue using AI

        Args:
            user_id: User identifier
            forecast_days: Number of days to forecast

        Returns:
            Revenue forecast with AI insights
        """
        try:
            # Get historical data for forecasting
            historical_data = self.get_business_intelligence(user_id, days=90)

            if historical_data["status"] != "success":
                return historical_data

            # Prepare forecast prompt
            forecast_prompt = f"""
            Based on the following historical financial data, forecast revenue for the next {forecast_days} days:

            Historical Data: {historical_data}

            Provide:
            1. Revenue forecast with confidence intervals
            2. Key drivers for the forecast
            3. Potential risks and opportunities
            4. Recommendations for revenue optimization
            """

            # Get AI forecast
            forecast_result = ai_service.process_natural_language_query(
                query=f"Forecast my business revenue for the next {forecast_days} days based on historical performance",
                data_schema={"revenue_metrics": "dict", "payment_stats": "dict", "financial_summary": "dict"},
                available_data=historical_data
            )

            return {
                "status": "success",
                "user_id": user_id,
                "forecast_period_days": forecast_days,
                "historical_basis": historical_data["financial_summary"],
                "forecast": forecast_result.get("response", "Forecast unavailable"),
                "confidence_level": "medium",  # Could be enhanced with actual confidence scoring
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to forecast revenue for user {user_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def _generate_business_recommendations(self, financial_data: Dict[str, Any]) -> List[str]:
        """Generate business recommendations based on financial data"""
        recommendations = []

        revenue = financial_data.get("total_revenue", 0)
        payments = financial_data.get("total_payments", 0)
        completion_rate = financial_data.get("payment_stats", {}).get("completion_rate", 0)

        if completion_rate < 0.95:
            recommendations.append("Payment completion rate is below 95%. Consider improving payment processing or customer experience.")

        if revenue > payments * 1.2:
            recommendations.append("Revenue significantly exceeds payment volume. Consider diversifying payment methods or marketing strategies.")

        if financial_data.get("transaction_count", 0) < 10:
            recommendations.append("Low transaction volume detected. Focus on customer acquisition and marketing campaigns.")

        avg_transaction = financial_data.get("average_transaction_value", 0)
        if avg_transaction > 1000:
            recommendations.append("High average transaction value indicates premium customer segment. Consider loyalty programs.")
        elif avg_transaction < 50:
            recommendations.append("Low average transaction value suggests price-sensitive market. Consider upselling strategies.")

        return recommendations if recommendations else ["Business metrics look healthy. Continue monitoring performance."]

# Global sync service instance
sync_service = SyncService()
