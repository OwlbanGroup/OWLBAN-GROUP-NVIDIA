#!/usr/bin/env python3
"""
OWLBAN GROUP AI - E2E Verification System
Simple end-to-end test to verify all systems work
"""

import sys
import json
import os
import ast

def main():
    results = {"passed": 0, "failed": 0, "tests": []}
    
    def add_result(name, passed, msg):
        results["tests"].append({"test": name, "passed": passed, "message": msg})
        if passed:
            results["passed"] += 1
            print(f"[PASS] {name}: {msg}")
        else:
            results["failed"] += 1
            print(f"[FAIL] {name}: {msg}")
    
    # Test Python version
    v = sys.version_info
    add_result("Python", v.major >= 3 and v.minor >= 8, f"{v.major}.{v.minor}.{v.micro}")
    
    # Test files exist
    for f in ["api_server.py", "database_manager.py", "auth_lib.py", "init.sql"]:
        add_result(f"File {f}", os.path.exists(f), "exists" if os.path.exists(f) else "missing")
    
    # Test database
    try:
        from database_manager import DatabaseManager
        db = DatabaseManager()
        status = db.get_database_status()
        ok = status.get("sqlite", {}).get("connected", False)
        add_result("Database", ok, "SQLite connected" if ok else "not connected")
        
        if ok:
            # Test CRUD
            db.add_employee("TEST-001", "Test", "User", "test@owlban.com", "Tester", "QA", 50000.0)
            emp = db.get_employee("TEST-001")
            add_result("Employee CRUD", emp is not None, "created and retrieved")
            db.delete_employee("TEST-001")
    except Exception as e:
        add_result("Database", False, str(e))
    
    # Test schema
    try:
        from database_manager import DatabaseManager
        db = DatabaseManager()
        cur = db.connections["sqlite"].cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        required = ['predictions', 'employees', 'employee_benefits', 'payroll']
        missing = [t for t in required if t not in tables]
        add_result("Schema", len(missing) == 0, f"tables: {len([t for t in required if t in tables])}/4")
    except Exception as e:
        add_result("Schema", False, str(e))
    
    # Test auth
    try:
        from auth_lib import authenticate_user, create_user, auth_manager
        create_user('e2e@test.com', 'e2e', 'TestPass123!', 'user', 'OWLBAN_GROUP')
        success, msg, user = authenticate_user('e2e@test.com', 'TestPass123!')
        add_result("Auth", bool(success and user is not None), msg)
        
        if success and user:
            at, rt = auth_manager.generate_tokens(user)
            payload = auth_manager.verify_access_token(at)
            add_result("Tokens", payload is not None, "JWT works")
    except Exception as e:
        add_result("Auth", False, str(e))
    
    # Test combined system
    try:
        from combined_nim_owlban_ai import CombinedSystem
        system = CombinedSystem()
        add_result("Combined System", True, "initialized")
    except Exception as e:
        add_result("Combined System", False, str(e))
    
    # Test quantum portfolio
    try:
        from quantum_financial_ai.quantum_portfolio_optimizer import QuantumPortfolioOptimizer
        opt = QuantumPortfolioOptimizer(use_gpu=False)
        
        class Asset:
            def __init__(self, sym, ret, vol, price, qty):
                self.symbol = sym; self.expected_return = ret
                self.volatility = vol; self.current_price = price; self.quantity = qty
        
        opt.add_asset(Asset('TEST', 0.12, 0.25, 150, 100))
        result = opt.optimize_portfolio(method="classical")
        add_result("Quantum Portfolio", result is not None, f"Sharpe: {result.sharpe_ratio:.4f}" if result else "failed")
    except Exception as e:
        add_result("Quantum Portfolio", False, str(e))
    
    # Test quantum risk
    try:
        from quantum_financial_ai.quantum_risk_analyzer import QuantumRiskAnalyzer, RiskFactor
        ana = QuantumRiskAnalyzer(use_gpu=False)
        ana.add_risk_factor(RiskFactor("MV", 0.15, 0.20))
        import numpy as np
        result = ana.analyze_risk(np.array([10000]), method="classical")
        add_result("Quantum Risk", result is not None, f"VaR: {result.value_at_risk:.2f}" if result else "failed")
    except Exception as e:
        add_result("Quantum Risk", False, str(e))
    
    # Test RL agent
    try:
        from performance_optimization.reinforcement_learning_agent import ReinforcementLearningAgent
        agent = ReinforcementLearningAgent(['a', 'b', 'c'])
        action = agent.choose_action([0.1, 0.2, 0.3, 0.4, 0.5])
        add_result("RL Agent", action in ['a', 'b', 'c'], f"action: {action}")
    except Exception as e:
        add_result("RL Agent", False, str(e))
    
    # Test anomaly detection
    try:
        from performance_optimization.advanced_anomaly_detection import AdvancedAnomalyDetection
        detector = AdvancedAnomalyDetection()
        import numpy as np
        data = np.array([100, 110, 105, 120, 200])
        result = detector.detect(data)
        add_result("Anomaly Detection", result is not None, "works")
    except Exception as e:
        add_result("Anomaly Detection", False, str(e))
    
    # Test API syntax
    try:
        with open("api_server.py") as f:
            ast.parse(f.read())
        add_result("API Syntax", True, "valid")
    except SyntaxError as e:
        add_result("API Syntax", False, f"line {e.lineno}")
    except Exception as e:
        add_result("API Syntax", False, str(e))
    
    # Summary
    print("\n" + "="*50)
    print(f"TESTS: {results['passed'] + results['failed']}")
    print(f"PASSED: {results['passed']}")
    print(f"FAILED: {results['failed']}")
    print("="*50)
    
    status = "ALL SYSTEMS OPERATIONAL" if results['failed'] == 0 else "NEEDS ATTENTION"
    print(f"STATUS: {status}")
    
    with open("e2e_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return 0 if results['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
