#!/usr/bin/env python3
"""
Comprehensive Blackbox AI Integration Test Suite
Tests the complete Blackbox AI integration for business operations
"""

import os
import sys
import json
import time
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import services
from ai_service import ai_service
from payments_service import payments_service
from revenue_service import revenue_service, RevenueType
from config import config
from logger import telemetry_logger

class BlackboxIntegrationTester:
    """Comprehensive tester for Blackbox AI business integration"""

    def __init__(self):
        self.logger = telemetry_logger.get_logger()
        self.test_results = []
        self.base_url = "http://localhost:5000"  # Assuming local test server

    def log_test_result(self, test_name: str, success: bool, message: str, details: Dict = None):
        """Log individual test result"""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details or {}
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        if details:
            print(f"   Details: {json.dumps(details, indent=2)}")

    def test_blackbox_ai_configuration(self):
        """Test 1: Verify Blackbox AI configuration"""
        try:
            # Check config settings
            blackbox_config = {
                'api_key_configured': bool(config.BLACKBOX_API_KEY),
                'base_url': config.BLACKBOX_BASE_URL,
                'model': config.BLACKBOX_MODEL,
                'temperature': config.BLACKBOX_TEMPERATURE
            }

            # Test AI service initialization
            ai_status = ai_service.get_service_status()

            success = ai_status.get('provider') == 'blackbox' and ai_status.get('status') == 'healthy'
            self.log_test_result(
                "Blackbox AI Configuration",
                success,
                f"Blackbox AI {'configured' if success else 'not configured'}",
                {
                    'config': blackbox_config,
                    'ai_service_status': ai_status
                }
            )
            return success
        except Exception as e:
            self.log_test_result("Blackbox AI Configuration", False, f"Configuration test failed: {str(e)}")
            return False

    def test_ai_business_analysis(self):
        """Test 2: Test AI-powered business analysis"""
        try:
            # Sample financial data
            financial_data = {
                'revenue': 150000,
                'expenses': 120000,
                'profit_margin': 0.2,
                'customer_count': 500,
                'monthly_growth': 0.15
            }

            # Test financial analysis
            analysis_result = ai_service.analyze_financial_data(
                financial_data,
                "Analyze this business performance and provide growth recommendations",
                "Q4 2024 business review"
            )

            success = analysis_result.get('status') == 'success'
            self.log_test_result(
                "AI Business Analysis",
                success,
                f"Financial analysis {'completed' if success else 'failed'}",
                {
                    'model_used': analysis_result.get('model_used'),
                    'analysis_length': len(analysis_result.get('analysis', ''))
                }
            )
            return success
        except Exception as e:
            self.log_test_result("AI Business Analysis", False, f"Business analysis test failed: {str(e)}")
            return False

    def test_ai_risk_assessment(self):
        """Test 3: Test AI-powered transaction risk assessment"""
        try:
            transaction_data = {
                'amount': 5000,
                'merchant': 'TechCorp Solutions',
                'category': 'software',
                'user_history': 'premium_customer',
                'location': 'domestic'
            }

            risk_result = ai_service.assess_transaction_risk(
                transaction_data,
                historical_patterns=[{'amount': 1000, 'approved': True}],
                market_conditions={'economic_stability': 'stable'}
            )

            success = risk_result.get('status') == 'success'
            self.log_test_result(
                "AI Risk Assessment",
                success,
                f"Risk assessment {'completed' if success else 'failed'}",
                {
                    'model_used': risk_result.get('model_used'),
                    'assessment_length': len(risk_result.get('risk_assessment', ''))
                }
            )
            return success
        except Exception as e:
            self.log_test_result("AI Risk Assessment", False, f"Risk assessment test failed: {str(e)}")
            return False

    def test_ai_natural_language_query(self):
        """Test 4: Test AI natural language business queries"""
        try:
            query = "What are the best strategies to increase revenue by 20% next quarter?"
            data_schema = {
                'revenue_table': ['amount', 'date', 'category'],
                'customer_table': ['id', 'segment', 'lifetime_value']
            }
            available_data = {
                'current_revenue': 150000,
                'customer_segments': ['enterprise', 'small_business', 'individual']
            }

            query_result = ai_service.process_natural_language_query(
                query, data_schema, available_data
            )

            success = query_result.get('status') == 'success'
            self.log_test_result(
                "AI Natural Language Query",
                success,
                f"Natural language query {'processed' if success else 'failed'}",
                {
                    'model_used': query_result.get('model_used'),
                    'response_length': len(query_result.get('response', ''))
                }
            )
            return success
        except Exception as e:
            self.log_test_result("AI Natural Language Query", False, f"Natural language query test failed: {str(e)}")
            return False

    def test_revenue_payment_sync(self):
        """Test 5: Test revenue and payment synchronization"""
        try:
            # Create a test revenue transaction
            revenue_tx = revenue_service.create_transaction(
                user_id='test_user_123',
                revenue_type=RevenueType.PURCHASE,
                amount=1000.0,
                description='Test purchase',
                merchant_name='Test Merchant'
            )

            # Create corresponding payment
            payment = payments_service.create_payment(
                amount=1000.0,
                payment_type=payments_service.PaymentType.CARD,
                user_id='test_user_123',
                description='Test payment for purchase'
            )

            # Verify sync
            revenue_record = revenue_service.get_transaction(revenue_tx.transaction_id)
            payment_record = payments_service.get_payment(payment.id)

            success = revenue_record is not None and payment_record is not None
            self.log_test_result(
                "Revenue-Payment Sync",
                success,
                f"Revenue-payment sync {'successful' if success else 'failed'}",
                {
                    'revenue_transaction': revenue_tx.transaction_id if revenue_tx else None,
                    'payment_transaction': payment.id if payment else None
                }
            )
            return success
        except Exception as e:
            self.log_test_result("Revenue-Payment Sync", False, f"Sync test failed: {str(e)}")
            return False

    def test_stripe_payment_integration(self):
        """Test 6: Test Stripe payment processing"""
        try:
            # Test payment intent creation (without real API call)
            if not config.STRIPE_SECRET_KEY:
                self.log_test_result("Stripe Payment Integration", False, "Stripe not configured")
                return False

            # Create payment intent
            intent_result = payments_service.create_stripe_payment_intent(
                amount=5000,  # $50.00
                currency='usd',
                description='Test business payment'
            )

            success = intent_result.get('status') == 'success'
            self.log_test_result(
                "Stripe Payment Integration",
                success,
                f"Stripe payment intent {'created' if success else 'failed'}",
                {
                    'intent_id': intent_result.get('payment_intent_id'),
                    'client_secret': intent_result.get('client_secret')[:10] + '...' if intent_result.get('client_secret') else None
                }
            )
            return success
        except Exception as e:
            self.log_test_result("Stripe Payment Integration", False, f"Stripe integration test failed: {str(e)}")
            return False

    def test_business_intelligence_endpoints(self):
        """Test 7: Test business intelligence API endpoints"""
        try:
            # Test AI analysis endpoint
            test_data = {
                'data': {'revenue': 100000, 'costs': 80000},
                'question': 'What is the profit margin?',
                'context': 'Monthly business review'
            }

            response = requests.post(
                f"{self.base_url}/ai/analyze",
                json=test_data,
                headers={'Authorization': 'Bearer test_token'},
                timeout=30
            )

            success = response.status_code == 200
            self.log_test_result(
                "Business Intelligence Endpoints",
                success,
                f"AI analysis endpoint {'accessible' if success else 'failed'}",
                {
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds()
                }
            )
            return success
        except requests.exceptions.RequestException as e:
            self.log_test_result("Business Intelligence Endpoints", False, f"API endpoint test failed: {str(e)}")
            return False

    def test_end_to_end_business_workflow(self):
        """Test 8: End-to-end business workflow test"""
        try:
            workflow_steps = []

            # Step 1: Create revenue transaction
            revenue_tx = revenue_service.create_transaction(
                user_id='e2e_test_user',
                revenue_type=RevenueType.PURCHASE,
                amount=2500.0,
                description='E2E test purchase'
            )
            workflow_steps.append({'step': 'revenue_creation', 'success': revenue_tx is not None})

            # Step 2: Process payment
            payment = payments_service.create_payment(
                amount=2500.0,
                payment_type=payments_service.PaymentType.CARD,
                user_id='e2e_test_user',
                description='E2E test payment'
            )
            workflow_steps.append({'step': 'payment_creation', 'success': payment is not None})

            # Step 3: Process payment
            payment_processed = payments_service.process_payment(payment.id)
            workflow_steps.append({'step': 'payment_processing', 'success': payment_processed})

            # Step 4: Process revenue transaction
            revenue_processed = revenue_service.process_transaction(revenue_tx.transaction_id, success=True)
            workflow_steps.append({'step': 'revenue_processing', 'success': revenue_processed})

            # Step 5: AI analysis of the transaction
            analysis_data = {
                'revenue': 2500.0,
                'transaction_type': 'purchase',
                'processed': True
            }
            ai_analysis = ai_service.analyze_financial_data(
                analysis_data,
                "Analyze this business transaction",
                "E2E workflow test"
            )
            workflow_steps.append({'step': 'ai_analysis', 'success': ai_analysis.get('status') == 'success'})

            # Overall success
            all_steps_success = all(step['success'] for step in workflow_steps)
            self.log_test_result(
                "End-to-End Business Workflow",
                all_steps_success,
                f"E2E workflow {'completed successfully' if all_steps_success else 'failed'}",
                {'workflow_steps': workflow_steps}
            )
            return all_steps_success
        except Exception as e:
            self.log_test_result("End-to-End Business Workflow", False, f"E2E workflow test failed: {str(e)}")
            return False

    def test_performance_and_scalability(self):
        """Test 9: Performance and scalability test"""
        try:
            start_time = time.time()

            # Test multiple AI queries in parallel
            queries = [
                "What are the best marketing strategies?",
                "How to optimize operational costs?",
                "What are market trends for our industry?",
                "How to improve customer retention?"
            ]

            results = []
            for query in queries:
                result = ai_service.process_natural_language_query(
                    query,
                    {'business_data': ['metrics', 'customers', 'operations']},
                    {'current_metrics': {'revenue': 100000}}
                )
                results.append(result)

            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / len(queries)

            # Performance criteria: average response time < 10 seconds
            success = avg_time < 10.0 and all(r.get('status') == 'success' for r in results)
            self.log_test_result(
                "Performance and Scalability",
                success,
                f"Performance test {'passed' if success else 'failed'}",
                {
                    'total_time': total_time,
                    'average_time': avg_time,
                    'queries_processed': len(results),
                    'all_successful': all(r.get('status') == 'success' for r in results)
                }
            )
            return success
        except Exception as e:
            self.log_test_result("Performance and Scalability", False, f"Performance test failed: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 Starting Blackbox AI Business Integration Tests")
        print("=" * 60)

        tests = [
            self.test_blackbox_ai_configuration,
            self.test_ai_business_analysis,
            self.test_ai_risk_assessment,
            self.test_ai_natural_language_query,
            self.test_revenue_payment_sync,
            self.test_stripe_payment_integration,
            self.test_business_intelligence_endpoints,
            self.test_end_to_end_business_workflow,
            self.test_performance_and_scalability
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            try:
                if test():
                    passed += 1
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                self.log_test_result(test.__name__, False, f"Test execution failed: {str(e)}")

        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed}/{total} tests passed")

        # Generate summary report
        self.generate_test_report()

        return passed == total

    def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'passed_tests': len([r for r in self.test_results if r['success']]),
                'failed_tests': len([r for r in self.test_results if not r['success']]),
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            'test_results': self.test_results,
            'system_info': {
                'blackbox_ai_configured': bool(config.BLACKBOX_API_KEY),
                'stripe_configured': bool(config.STRIPE_SECRET_KEY),
                'ai_service_provider': ai_service.get_service_status().get('provider')
            }
        }

        # Save report to file
        report_file = f"blackbox_integration_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Detailed test report saved to: {report_file}")

        # Print summary
        print("\n🎯 Blackbox AI Integration Status:")
        if report['test_summary']['passed_tests'] == report['test_summary']['total_tests']:
            print("✅ ALL TESTS PASSED - Blackbox AI business integration is fully operational!")
        else:
            print("⚠️  SOME TESTS FAILED - Review the detailed report for issues")

        return report

def main():
    """Main test execution"""
    tester = BlackboxIntegrationTester()

    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
