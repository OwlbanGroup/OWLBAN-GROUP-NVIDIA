#!/usr/bin/env python3
"""
JPMorgan Financial APIs - Interactive Demo Script
===============================================

This script demonstrates how to use the JPMorgan Financial APIs platform,
showcasing all major features including authentication, business management,
revenue tracking, audit logging, ML anomaly detection, and real-time features.

Requirements:
- Python 3.8+
- requests library (pip install requests)
- websocket-client library (pip install websocket-client)
- The JPMorgan API server running on localhost:5000

Usage:
    python demo_script.py

Author: JPMorgan Financial APIs Team
"""

import json
import time
import requests
import websocket
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class JPMorganAPIDemo:
    """Interactive demo for JPMorgan Financial APIs"""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.auth_token = None
        self.websocket_connected = False

        print("🚀 JPMorgan Financial APIs - Interactive Demo")
        print("=" * 50)

    def make_request(self, method: str, endpoint: str, data: Dict = None,
                    headers: Dict = None, auth_required: bool = True) -> Dict:
        """Make HTTP request with proper error handling"""
        url = f"{self.base_url}{endpoint}"

        request_headers = {'Content-Type': 'application/json'}
        if headers:
            request_headers.update(headers)

        if auth_required and self.auth_token:
            request_headers['Authorization'] = f'Bearer {self.auth_token}'

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, headers=request_headers, params=data)
            elif method.upper() == 'POST':
                response = self.session.post(url, headers=request_headers, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, headers=request_headers, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, headers=request_headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            print(f"\n📡 {method.upper()} {endpoint} -> {response.status_code}")

            if response.status_code >= 400:
                print(f"❌ Error: {response.text}")
                return None

            try:
                return response.json()
            except:
                return {'message': response.text}

        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None

    def demo_health_check(self):
        """Demo: Health check endpoint"""
        print("\n🏥 1. HEALTH CHECK")
        print("-" * 30)

        result = self.make_request('GET', '/health', auth_required=False)
        if result:
            print("✅ API is healthy!")
            print(f"   Version: {result.get('version', 'Unknown')}")
            print(f"   Timestamp: {result.get('timestamp', 'Unknown')}")

    def demo_user_registration(self):
        """Demo: User registration"""
        print("\n👤 2. USER REGISTRATION")
        print("-" * 30)

        # Register a new user
        user_data = {
            'username': f'demo_user_{int(time.time())}',
            'password': 'DemoPass123!'
        }

        result = self.make_request('POST', '/user/register', user_data, auth_required=False)
        if result and result.get('status') == 'success':
            print("✅ User registered successfully!")
            print(f"   Username: {user_data['username']}")
            return user_data['username']
        else:
            print("❌ User registration failed")
            return None

    def demo_user_login(self, username: str):
        """Demo: User login"""
        print("\n🔐 3. USER LOGIN")
        print("-" * 30)

        login_data = {
            'username': username,
            'password': 'DemoPass123!'
        }

        result = self.make_request('POST', '/user/login', login_data, auth_required=False)
        if result and result.get('status') == 'success':
            self.auth_token = result.get('token')
            print("✅ Login successful!")
            print(f"   Token: {self.auth_token[:20]}...")
            return True
        else:
            print("❌ Login failed")
            return False

    def demo_user_profile(self):
        """Demo: Get user profile"""
        print("\n👤 4. USER PROFILE")
        print("-" * 30)

        result = self.make_request('GET', '/user/profile')
        if result and result.get('status') == 'success':
            print("✅ Profile retrieved!")
            print(f"   Username: {result.get('username')}")
            print(f"   Created: {result.get('created_at')}")

    def demo_business_management(self):
        """Demo: Business CRUD operations"""
        print("\n🏢 5. BUSINESS MANAGEMENT")
        print("-" * 30)

        # Create a business
        business_data = {
            'name': 'Demo Tech Solutions Inc.',
            'type': 'corporation',
            'registration_number': '123456789',
            'address': '123 Business Ave, New York, NY 10001',
            'contact_info': {
                'email': 'contact@demotech.com',
                'phone': '+1-555-0123',
                'website': 'https://demotech.com'
            }
        }

        print("Creating business...")
        result = self.make_request('POST', '/businesses', business_data)
        if result and result.get('status') == 'success':
            business = result.get('business', {})
            business_id = business.get('id')
            print("✅ Business created!")
            print(f"   ID: {business_id}")
            print(f"   Name: {business.get('name')}")

            # List all businesses
            print("\nListing all businesses...")
            result = self.make_request('GET', '/businesses')
            if result and result.get('status') == 'success':
                print(f"✅ Found {result.get('count', 0)} businesses")

            # Get specific business
            print(f"\nGetting business {business_id}...")
            result = self.make_request('GET', f'/businesses/{business_id}')
            if result and result.get('status') == 'success':
                print("✅ Business details retrieved!")

            # Update business
            update_data = {
                'name': 'Demo Tech Solutions Inc. (Updated)',
                'address': '456 Updated Ave, New York, NY 10002'
            }
            print(f"\nUpdating business {business_id}...")
            result = self.make_request('PUT', f'/businesses/{business_id}', update_data)
            if result and result.get('status') == 'success':
                print("✅ Business updated!")

            return business_id
        return None

    def demo_asset_management(self, business_id: int):
        """Demo: Asset CRUD operations"""
        print("\n💼 6. ASSET MANAGEMENT")
        print("-" * 30)

        # Create an asset
        asset_data = {
            'business_id': business_id,
            'name': 'Office Building - Manhattan',
            'type': 'real_estate',
            'value': 25000000.00,
            'acquisition_date': '2023-01-15T00:00:00Z',
            'description': 'Premium office space in Manhattan'
        }

        print("Creating asset...")
        result = self.make_request('POST', '/assets', asset_data)
        if result and result.get('status') == 'success':
            asset = result.get('asset', {})
            asset_id = asset.get('id')
            print("✅ Asset created!")
            print(f"   ID: {asset_id}")
            print(f"   Name: {asset.get('name')}")
            print(f"   Value: ${asset.get('value', 0):.2f}")
            # List all assets
            print("\nListing all assets...")
            result = self.make_request('GET', '/assets')
            if result and result.get('status') == 'success':
                print(f"✅ Found {result.get('count', 0)} assets")

            # Get business assets
            print(f"\nGetting assets for business {business_id}...")
            result = self.make_request('GET', f'/businesses/{business_id}/assets')
            if result and result.get('status') == 'success':
                print(f"✅ Found {result.get('count', 0)} assets for this business")

            return asset_id
        return None

    def demo_revenue_tracking(self):
        """Demo: Revenue transaction processing"""
        print("\n💰 7. REVENUE TRACKING")
        print("-" * 30)

        # Create a revenue transaction
        transaction_data = {
            'user_id': 'demo_user_123',
            'revenue_type': 'purchase',
            'amount': 1500.00,
            'currency': 'USD',
            'description': 'Software license purchase',
            'merchant_name': 'Demo Software Corp',
            'category': 'Software',
            'payment_method': 'credit_card'
        }

        print("Creating revenue transaction...")
        result = self.make_request('POST', '/revenue/transactions', transaction_data)
        if result and result.get('status') == 'success':
            transaction = result.get('transaction', {})
            transaction_id = transaction.get('transaction_id')
            print("✅ Transaction created!")
            print(f"   ID: {transaction_id}")
            print(f"   Amount: ${transaction.get('amount', 0):.2f}")
            print(f"   Net Amount: ${transaction.get('net_amount', 0):.2f}")

            # Get transaction details
            print(f"\nGetting transaction {transaction_id}...")
            result = self.make_request('GET', f'/revenue/transactions/{transaction_id}')
            if result and result.get('status') == 'success':
                print("✅ Transaction details retrieved!")

            # Process the transaction
            process_data = {'success': True}
            print(f"\nProcessing transaction {transaction_id}...")
            result = self.make_request('POST', f'/revenue/transactions/{transaction_id}/process', process_data)
            if result and result.get('status') == 'success':
                print("✅ Transaction processed successfully!")

            # Get revenue metrics
            print("\nGetting revenue metrics...")
            metrics_data = {
                'start_date': '2024-01-01T00:00:00Z',
                'end_date': '2024-12-31T23:59:59Z'
            }
            result = self.make_request('GET', '/revenue/metrics', metrics_data)
            if result and result.get('status') == 'success':
                metrics = result.get('metrics', {})
                print("✅ Revenue metrics retrieved!")
                print(f"   Total Revenue: ${metrics.get('total_amount', 0):.2f}")
                print(f"   Transaction Count: {metrics.get('transaction_count', 0)}")

            return transaction_id
        return None

    def demo_telemetry_processing(self):
        """Demo: Telemetry data processing"""
        print("\n📊 8. TELEMETRY PROCESSING")
        print("-" * 30)

        # Sample telemetry data
        telemetry_data = {
            'name': 'Microsoft.WindowsStore.8wekyb3d8bbwe',
            'ver': '12101.1001.1.0',
            'data': {
                'Op': 'Purchase',
                'PFN': 'Microsoft.WindowsStore_8wekyb3d8bbwe',
                'OS': 'Windows 11 Pro',
                'DeviceModel': 'Surface Pro 9',
                'UserId': 'demo_user_123',
                'SessionId': 'session_abc123',
                'Timestamp': datetime.now(timezone.utc).isoformat(),
                'EventType': 'Purchase',
                'Amount': 29.99,
                'Currency': 'USD',
                'ProductId': '9WZDNCRFJ364',
                'Category': 'Entertainment'
            }
        }

        print("Processing telemetry data...")
        result = self.make_request('POST', '/telemetry', telemetry_data)
        if result and result.get('status') == 'success':
            print("✅ Telemetry data processed!")

            # Get telemetry metrics
            print("\nGetting telemetry metrics...")
            result = self.make_request('GET', '/telemetry/metrics?hours=24')
            if result and result.get('status') == 'success':
                metrics = result.get('metrics', {})
                print("✅ Telemetry metrics retrieved!")
                print(f"   Events processed: {metrics.get('total_events', 0)}")
                print(f"   Data points: {metrics.get('total_data_points', 0)}")

    def demo_ml_anomaly_detection(self):
        """Demo: ML anomaly detection"""
        print("\n🤖 9. ML ANOMALY DETECTION")
        print("-" * 30)

        # Sample telemetry data for anomaly detection
        batch_data = {
            'telemetry_data': [
                {
                    'name': 'Microsoft.WindowsStore.8wekyb3d8bbwe',
                    'ver': '12101.1001.1.0',
                    'data': {
                        'Op': 'Purchase',
                        'Amount': 29.99,
                        'UserId': 'user_123'
                    }
                },
                {
                    'name': 'Microsoft.WindowsStore.8wekyb3d8bbwe',
                    'ver': '12101.1001.1.0',
                    'data': {
                        'Op': 'Purchase',
                        'Amount': 999.99,  # Anomalous high amount
                        'UserId': 'user_456'
                    }
                }
            ]
        }

        print("Running anomaly detection...")
        result = self.make_request('POST', '/ml/anomalies', batch_data)
        if result and result.get('status') == 'success':
            anomalies = result.get('anomaly_results', [])
            print("✅ Anomaly detection completed!")
            print(f"   Anomalies found: {len([a for a in anomalies if a.get('is_anomaly', False)])}")

            # Train ML model
            training_data = {
                'training_data': [
                    [29.99, 1, 1, 1, 1, 1, 1],
                    [49.99, 1, 1, 1, 1, 1, 1],
                    [999.99, 1, 1, 1, 1, 1, 1],  # Anomalous
                    [19.99, 1, 1, 1, 1, 1, 1]
                ],
                'contamination': 0.1
            }

            print("\nTraining ML model...")
            result = self.make_request('POST', '/ml/train', training_data)
            if result and result.get('status') == 'success':
                print("✅ ML model trained successfully!")

    def demo_private_banking(self):
        """Demo: Private banking services"""
        print("\n🏦 10. PRIVATE BANKING SERVICES")
        print("-" * 30)

        # Get private bank accounts
        print("Getting private bank accounts...")
        result = self.make_request('GET', '/private-bank/accounts')
        if result and result.get('status') == 'success':
            accounts = result.get('accounts', [])
            print(f"✅ Found {len(accounts)} private bank accounts")

            if accounts:
                account = accounts[0]
                print(f"   Sample Account: {account.get('account_id')} - ${account.get('balance', 0):,.2f}")

        # Get wealth management portfolio
        print("\nGetting wealth management portfolio...")
        result = self.make_request('GET', '/private-bank/wealth')
        if result and result.get('status') == 'success':
            portfolio = result.get('portfolio', {})
            print("✅ Wealth portfolio retrieved!")
            print(f"   Total Value: ${portfolio.get('total_value', 0):,.2f}")
        # Get investment portfolio
        print("\nGetting investment portfolio...")
        result = self.make_request('GET', '/private-bank/investments')
        if result and result.get('status') == 'success':
            investments = result.get('investments', [])
            print(f"✅ Found {len(investments)} investments")

    def demo_audit_logging(self):
        """Demo: Audit logging and compliance"""
        print("\n🔒 11. AUDIT LOGGING & COMPLIANCE")
        print("-" * 30)

        # Get audit logs
        print("Getting audit logs...")
        result = self.make_request('GET', '/audit/logs?limit=10')
        if result and result.get('status') == 'success':
            logs = result.get('logs', [])
            print(f"✅ Retrieved {len(logs)} audit log entries")

            if logs:
                log = logs[0]
                print(f"   Sample Log: {log.get('action')} by {log.get('username', 'Unknown')}")

        # Get audit summary
        print("\nGetting audit summary...")
        result = self.make_request('GET', '/audit/summary')
        if result and result.get('status') == 'success':
            summary = result.get('summary', {})
            print("✅ Audit summary retrieved!")
            print(f"   Total Events: {summary.get('total_events', 0)}")
            print(f"   Security Events: {summary.get('security_events', 0)}")

        # Get security report
        print("\nGetting security report...")
        result = self.make_request('GET', '/audit/reports/security')
        if result and result.get('status') == 'success':
            report = result.get('report', {})
            print("✅ Security report generated!")
            print(f"   Failed Logins: {report.get('failed_login_attempts', 0)}")
            print(f"   Suspicious Activities: {report.get('suspicious_activities', 0)}")

    def demo_jpmorgan_data(self):
        """Demo: JPMorgan financial data"""
        print("\n🏦 12. JPMORGAN FINANCIAL DATA")
        print("-" * 30)

        result = self.make_request('GET', '/api/jpmorgan-data')
        if result and result.get('status') == 'success':
            financial_metrics = result.get('financial_metrics', {})
            stock_ticker = result.get('stock_ticker', {})
            assets = result.get('assets', [])

            print("✅ JPMorgan financial data retrieved!")
            print(f"   Market Cap: ${financial_metrics.get('market_cap', 0):,.2f}")
            print(f"   Revenue: ${financial_metrics.get('revenue', 0):,.2f}")
            print(f"   Stock Price: ${stock_ticker.get('current_price', 0):.2f}")
            print(f"   Assets: {len(assets)} major holdings")

    def demo_websocket_connection(self):
        """Demo: WebSocket real-time features"""
        print("\n🔄 13. WEBSOCKET REAL-TIME FEATURES")
        print("-" * 30)

        try:
            import websocket
            print("WebSocket demo requires manual testing with the web dashboard")
            print("Please open http://localhost:5000/dashboard in your browser")
            print("to see real-time updates and WebSocket functionality")

        except ImportError:
            print("websocket-client library not installed")
            print("Install with: pip install websocket-client")

    def demo_data_conversion(self):
        """Demo: Data format conversion"""
        print("\n🔄 14. DATA FORMAT CONVERSION")
        print("-" * 30)

        # Sample data for conversion
        sample_data = [
            {'name': 'John Doe', 'age': 30, 'city': 'New York'},
            {'name': 'Jane Smith', 'age': 25, 'city': 'Los Angeles'}
        ]

        conversion_data = {
            'data': sample_data,
            'from_format': 'json',
            'to_format': 'csv'
        }

        print("Converting JSON to CSV...")
        result = self.make_request('POST', '/data/convert', conversion_data, auth_required=False)
        if result:
            print("✅ Data conversion successful!")
            print("CSV Output:")
            print(result)

    def run_full_demo(self):
        """Run the complete API demonstration"""
        print("\n🎬 STARTING JPMORGAN FINANCIAL APIs DEMO")
        print("=" * 50)

        # Basic health check
        self.demo_health_check()

        # User management
        username = self.demo_user_registration()
        if not username:
            print("❌ Cannot continue without user registration")
            return

        if not self.demo_user_login(username):
            print("❌ Cannot continue without login")
            return

        self.demo_user_profile()

        # Business and asset management
        business_id = self.demo_business_management()
        if business_id:
            self.demo_asset_management(business_id)

        # Revenue and financial operations
        self.demo_revenue_tracking()
        self.demo_telemetry_processing()
        self.demo_ml_anomaly_detection()

        # Banking services
        self.demo_private_banking()
        self.demo_jpmorgan_data()

        # Compliance and security
        self.demo_audit_logging()

        # Utilities
        self.demo_data_conversion()
        self.demo_websocket_connection()

        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("The JPMorgan Financial APIs platform includes:")
        print("• 28 production-ready endpoints")
        print("• Enterprise-grade security and compliance")
        print("• Real-time telemetry processing")
        print("• ML-powered anomaly detection")
        print("• Comprehensive audit logging")
        print("• WebSocket real-time updates")
        print("• Full business and revenue management")
        print("\nFor more information, visit: http://localhost:5000/")


def main():
    """Main demo function"""
    demo = JPMorganAPIDemo()

    try:
        demo.run_full_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Make sure the JPMorgan API server is running on localhost:5000")


if __name__ == "__main__":
    main()
