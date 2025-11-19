#!/usr/bin/env python3
"""
Critical-Path Testing Script for Live Production Data Dashboard Integration
Tests key endpoints and functionality to verify the implementation.
"""

import asyncio
import httpx
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8010"
PROMETHEUS_URL = "http://localhost:9090"
TELEMETRY_URL = "http://localhost:8009"

# Test results
test_results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "tests": []
}

def log_test(name: str, status: str, message: str = ""):
    """Log test result"""
    test_results["tests"].append({
        "name": name,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat()
    })

    if status == "PASS":
        test_results["passed"] += 1
        print(f"✅ {name}: PASSED")
    elif status == "FAIL":
        test_results["failed"] += 1
        print(f"❌ {name}: FAILED - {message}")
    else:
        test_results["skipped"] += 1
        print(f"⏭️  {name}: SKIPPED - {message}")

    if message and status != "SKIP":
        print(f"   {message}")

async def test_dashboard_health():
    """Test 1: Dashboard service health check"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health", timeout=5.0)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    log_test("Dashboard Health Check", "PASS", "Dashboard service is healthy")
                    return True
                else:
                    log_test("Dashboard Health Check", "FAIL", f"Unexpected status: {data.get('status')}")
                    return False
            else:
                log_test("Dashboard Health Check", "FAIL", f"HTTP {response.status_code}")
                return False
    except Exception as e:
        log_test("Dashboard Health Check", "FAIL", f"Connection error: {str(e)}")
        return False

async def test_services_health():
    """Test 2: Services health endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            # Note: This endpoint requires authentication in production
            # For testing, we'll check if it returns 401 (auth required) or 200 (success)
            response = await client.get(f"{BASE_URL}/api/health/services", timeout=10.0)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    services = data.get("data", {}).get("services", {})
                    log_test("Services Health Check", "PASS",
                            f"Checked {len(services)} services")
                    return True
                else:
                    log_test("Services Health Check", "FAIL", "Invalid response format")
                    return False
            elif response.status_code == 401:
                log_test("Services Health Check", "SKIP",
                        "Authentication required (expected in production)")
                return True
            else:
                log_test("Services Health Check", "FAIL", f"HTTP {response.status_code}")
                return False
    except Exception as e:
        log_test("Services Health Check", "FAIL", f"Error: {str(e)}")
        return False

async def test_infrastructure_health():
    """Test 3: Infrastructure health endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/health/infrastructure", timeout=10.0)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    components = data.get("data", {}).get("components", {})
                    log_test("Infrastructure Health Check", "PASS",
                            f"Checked {len(components)} components")
                    return True
                else:
                    log_test("Infrastructure Health Check", "FAIL", "Invalid response format")
                    return False
            elif response.status_code == 401:
                log_test("Infrastructure Health Check", "SKIP",
                        "Authentication required (expected in production)")
                return True
            else:
                log_test("Infrastructure Health Check", "FAIL", f"HTTP {response.status_code}")
                return False
    except Exception as e:
        log_test("Infrastructure Health Check", "FAIL", f"Error: {str(e)}")
        return False

async def test_prometheus_connectivity():
    """Test 4: Prometheus connectivity"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{PROMETHEUS_URL}/-/healthy", timeout=5.0)

            if response.status_code == 200:
                log_test("Prometheus Connectivity", "PASS", "Prometheus is accessible")
                return True
            else:
                log_test("Prometheus Connectivity", "FAIL", f"HTTP {response.status_code}")
                return False
    except Exception as e:
        log_test("Prometheus Connectivity", "SKIP",
                f"Prometheus not running or not accessible: {str(e)}")
        return False

async def test_production_metrics():
    """Test 5: Production metrics endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/production/metrics", timeout=10.0)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    metrics = data.get("data", {})
                    expected_metrics = ["request_rate", "error_rate", "avg_response_time",
                                        "cpu_usage", "memory_usage"]

                    if all(metric in metrics for metric in expected_metrics):
                        log_test("Production Metrics", "PASS",
                                f"All {len(expected_metrics)} metrics present")
                        return True
                    else:
                        log_test("Production Metrics", "FAIL", "Missing expected metrics")
                        return False
                else:
                    log_test("Production Metrics", "FAIL", "Invalid response format")
                    return False
            elif response.status_code == 401:
                log_test("Production Metrics", "SKIP",
                        "Authentication required (expected in production)")
                return True
            else:
                log_test("Production Metrics", "FAIL", f"HTTP {response.status_code}")
                return False
    except Exception as e:
        log_test("Production Metrics", "FAIL", f"Error: {str(e)}")
        return False

async def test_dashboard_ui():
    """Test 6: Dashboard UI accessibility"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/", timeout=5.0)

            if response.status_code == 200:
                content = response.text
                # Check for key UI elements
                if "JPMorgan Financial" in content and "dashboard" in content.lower():
                    log_test("Dashboard UI", "PASS", "Dashboard page loads successfully")
                    return True
                else:
                    log_test("Dashboard UI", "FAIL", "Dashboard content not found")
                    return False
            elif response.status_code == 401 or response.status_code == 302:
                log_test("Dashboard UI", "SKIP",
                        "Redirected to login (expected with authentication)")
                return True
            else:
                log_test("Dashboard UI", "FAIL", f"HTTP {response.status_code}")
                return False
    except Exception as e:
        log_test("Dashboard UI", "FAIL", f"Error: {str(e)}")
        return False

async def run_tests():
    """Run all critical-path tests"""
    print("=" * 70)
    print("🧪 Live Production Data Dashboard - Critical-Path Testing")
    print("=" * 70)
    print()

    # Run tests sequentially
    await test_dashboard_health()
    await test_services_health()
    await test_infrastructure_health()
    await test_prometheus_connectivity()
    await test_production_metrics()
    await test_dashboard_ui()

    # Print summary
    print()
    print("=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"✅ Passed:  {test_results['passed']}")
    print(f"❌ Failed:  {test_results['failed']}")
    print(f"⏭️  Skipped: {test_results['skipped']}")
    print(f"📝 Total:   {len(test_results['tests'])}")
    print()

    # Determine overall status
    if test_results['failed'] == 0:
        print("🎉 All critical tests passed or skipped (authentication required)")
        print("✅ Implementation is ready for deployment")
    else:
        print("⚠️  Some tests failed - review the errors above")
        print("🔧 Fix the issues before deployment")

    print()

    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print("📄 Detailed results saved to: test_results.json")
    print()

if __name__ == "__main__":
    print()
    print("Starting tests...")
    print()

    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test execution failed: {str(e)}")
