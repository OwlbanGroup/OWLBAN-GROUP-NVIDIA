"""
Payroll Service Module for JPMorgan Financial APIs
Provides payroll processing functionality.
"""

import secrets
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

try:
    from src.logger import telemetry_logger
except ImportError:
    class FallbackLogger:
        def log_info(self, msg, context=None):
            print(f"INFO: {msg}")
        def log_error(self, msg, context=None):
            print(f"ERROR: {msg}")
    telemetry_logger = FallbackLogger()


# =============================================================================
# IN-MOCK DATA STORE (Replace with database in production)
# =============================================================================

class InMemoryPayrollStore:
    """In-memory storage for payroll data (replace with database)"""
    
    def __init__(self):
        self.employees = {}
        self.payroll_runs = {}
        self.payments = {}
    
    def add_employee(self, employee: Dict[str, Any]) -> Dict[str, Any]:
        """Add an employee"""
        employee_id = employee['employee_id']
        self.employees[employee_id] = employee
        return employee
    
    def get_employee(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Get an employee by ID"""
        return self.employees.get(employee_id)
    
    def get_employees_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all employees for a user"""
        return [e for e in self.employees.values() if e.get('user_id') == user_id]
    
    def update_employee(self, employee_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an employee"""
        if employee_id in self.employees:
            self.employees[employee_id].update(updates)
            return self.employees[employee_id]
        return None
    
    def delete_employee(self, employee_id: str) -> bool:
        """Delete an employee"""
        if employee_id in self.employees:
            del self.employees[employee_id]
            return True
        return False
    
    def add_payroll_run(self, run: Dict[str, Any]) -> Dict[str, Any]:
        """Add a payroll run"""
        run_id = run['run_id']
        self.payroll_runs[run_id] = run
        return run
    
    def get_payroll_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a payroll run by ID"""
        return self.payroll_runs.get(run_id)
    
    def get_payroll_runs_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all payroll runs for a user"""
        return [r for r in self.payroll_runs.values() if r.get('user_id') == user_id]
    
    def update_payroll_run(self, run_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a payroll run"""
        if run_id in self.payroll_runs:
            self.payroll_runs[run_id].update(updates)
            return self.payroll_runs[run_id]
        return None
    
    def add_payment(self, payment: Dict[str, Any]) -> Dict[str, Any]:
        """Add a payroll payment"""
        payment_id = payment['payment_id']
        self.payments[payment_id] = payment
        return payment
    
    def get_payment(self, payment_id: str) -> Optional[Dict[str, Any]]:
        """Get a payment by ID"""
        return self.payments.get(payment_id)
    
    def get_payments_by_employee(self, employee_id: str) -> List[Dict[str, Any]]:
        """Get all payments for an employee"""
        return [p for p in self.payments.values() if p.get('employee_id') == employee_id]
    
    def get_payments_by_run(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all payments for a payroll run"""
        return [p for p in self.payments.values() if p.get('payroll_run_id') == run_id]


# Global store instance
payroll_store = InMemoryPayrollStore()


# =============================================================================
# TAX CALCULATOR
# =============================================================================

class TaxCalculator:
    """Tax calculation utilities"""
    
    # Federal tax brackets (simplified 2024)
    FEDERAL_TAX_BRACKETS = [
        (11600, 0.10),
        (47150, 0.12),
        (100525, 0.22),
        (191950, 0.24),
        (243725, 0.32),
        (609350, 0.35),
        (float('inf'), 0.37)
    ]
    
    # Social Security rate
    SOCIAL_SECURITY_RATE = 0.062
    SOCIAL_SECURITY_WAGE_BASE = 168600
    
    # Medicare rate
    MEDICARE_RATE = 0.0145
    MEDICARE_ADDITIONAL_RATE = 0.009
    MEDICARE_ADDITIONAL_THRESHOLD = 200000
    
    @staticmethod
    def calculate_federal_tax(annual_gross: float) -> float:
        """Calculate federal income tax"""
        tax = 0.0
        remaining = annual_gross
        previous_threshold = 0
        
        for threshold, rate in TaxCalculator.FEDERAL_TAX_BRACKETS:
            if remaining <= 0:
                break
            taxable_in_bracket = min(remaining, threshold - previous_threshold)
            tax += taxable_in_bracket * rate
            remaining -= taxable_in_bracket
            previous_threshold = threshold
        
        return tax
    
    @staticmethod
    def calculate_state_tax(annual_gross: float, state_rate: float = 0.05) -> float:
        """Calculate state income tax (simplified)"""
        return annual_gross * state_rate
    
    @staticmethod
    def calculate_social_security(annual_gross: float) -> float:
        """Calculate Social Security tax"""
        taxable_wages = min(annual_gross, TaxCalculator.SOCIAL_SECURITY_WAGE_BASE)
        return taxable_wages * TaxCalculator.SOCIAL_SECURITY_RATE
    
    @staticmethod
    def calculate_medicare(annual_gross: float) -> float:
        """Calculate Medicare tax"""
        base_tax = annual_gross * TaxCalculator.MEDICARE_RATE
        
        # Additional Medicare tax
        if annual_gross > TaxCalculator.MEDICARE_ADDITIONAL_THRESHOLD:
            additional_tax = (annual_gross - TaxCalculator.MEDICARE_ADDITIONAL_THRESHOLD) * TaxCalculator.MEDICARE_ADDITIONAL_RATE
            base_tax += additional_tax
        
        return base_tax
    
    @staticmethod
    def calculate_all_taxes(annual_gross: float, state_rate: float = 0.05) -> Dict[str, float]:
        """Calculate all taxes"""
        return {
            'federal_tax': TaxCalculator.calculate_federal_tax(annual_gross),
            'state_tax': TaxCalculator.calculate_state_tax(annual_gross, state_rate),
            'social_security': TaxCalculator.calculate_social_security(annual_gross),
            'medicare': TaxCalculator.calculate_medicare(annual_gross)
        }


# =============================================================================
# PAYROLL SERVICE
# =============================================================================

class PayrollService:
    """Service for managing payroll"""
    
    def __init__(self):
        self.store = payroll_store
        self.tax_calculator = TaxCalculator()
        self.logger = telemetry_logger
    
    # Employee Management
    def create_employee(self, user_id: str, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new employee"""
        
        # Validate required fields
        required_fields = ['first_name', 'last_name', 'email']
        for field in required_fields:
            if field not in employee_data:
                return {
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }
        
        # Generate employee ID
        employee_id = employee_data.get('employee_id') or f"EMP-{secrets.token_hex(4).upper()}"
        
        # Check for duplicate
        if self.store.get_employee(employee_id):
            return {
                'status': 'error',
                'message': f'Employee with ID {employee_id} already exists'
            }
        
        employee = {
            'employee_id': employee_id,
            'user_id': user_id,
            'first_name': employee_data['first_name'],
            'last_name': employee_data['last_name'],
            'email': employee_data['email'],
            'phone': employee_data.get('phone'),
            'department': employee_data.get('department'),
            'position': employee_data.get('position'),
            'hire_date': employee_data.get('hire_date', datetime.now(timezone.utc).isoformat()),
            'employment_type': employee_data.get('employment_type', 'full_time'),
            'salary': employee_data.get('salary', 0),
            'hourly_rate': employee_data.get('hourly_rate'),
            'pay_frequency': employee_data.get('pay_frequency', 'biweekly'),
            'tax_filing_status': employee_data.get('tax_filing_status', 'single'),
            'tax_withholding_rate': employee_data.get('tax_withholding_rate', 0.20),
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        self.store.add_employee(employee)
        
        self.logger.log_info(f"Employee created: {employee_id}", {'context': 'payroll_service'})
        
        return {
            'status': 'success',
            'employee': employee
        }
    
    def get_employee(self, employee_id: str) -> Dict[str, Any]:
        """Get an employee by ID"""
        employee = self.store.get_employee(employee_id)
        
        if not employee:
            return {
                'status': 'error',
                'message': 'Employee not found'
            }
        
        return {
            'status': 'success',
            'employee': employee
        }
    
    def list_employees(self, user_id: str) -> Dict[str, Any]:
        """List all employees for a user"""
        employees = self.store.get_employees_by_user(user_id)
        
        return {
            'status': 'success',
            'employees': employees,
            'count': len(employees)
        }
    
    def update_employee(self, employee_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an employee"""
        employee = self.store.update_employee(employee_id, updates)
        
        if not employee:
            return {
                'status': 'error',
                'message': 'Employee not found'
            }
        
        employee['updated_at'] = datetime.now(timezone.utc).isoformat()
        
        return {
            'status': 'success',
            'employee': employee
        }
    
    def delete_employee(self, employee_id: str) -> Dict[str, Any]:
        """Delete an employee"""
        success = self.store.delete_employee(employee_id)
        
        if not success:
            return {
                'status': 'error',
                'message': 'Employee not found'
            }
        
        return {
            'status': 'success',
            'message': 'Employee deleted successfully'
        }
    
    # Payroll Run Management
    def create_payroll_run(self, user_id: str, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new payroll run"""
        
        required_fields = ['pay_period_start', 'pay_period_end', 'payment_date']
        for field in required_fields:
            if field not in run_data:
                return {
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }
        
        # Generate run ID
        run_id = f"PR-{secrets.token_hex(4).upper()}"
        
        run = {
            'run_id': run_id,
            'user_id': user_id,
            'pay_period_start': run_data['pay_period_start'],
            'pay_period_end': run_data['pay_period_end'],
            'payment_date': run_data['payment_date'],
            'total_gross_pay': 0,
            'total_net_pay': 0,
            'total_deductions': 0,
            'total_taxes': 0,
            'employee_count': 0,
            'status': 'pending',
            'initiated_at': datetime.now(timezone.utc).isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        self.store.add_payroll_run(run)
        
        return {
            'status': 'success',
            'run': run
        }
    
    def process_payroll_run(self, run_id: str) -> Dict[str, Any]:
        """Process a payroll run - calculate payments for all employees"""
        
        run = self.store.get_payroll_run(run_id)
        if not run:
            return {
                'status': 'error',
                'message': 'Payroll run not found'
            }
        
        if run['status'] != 'pending':
            return {
                'status': 'error',
                'message': f'Cannot process payroll run with status: {run["status"]}'
            }
        
        # Get employees for this user
        employees = self.store.get_employees_by_user(run['user_id'])
        
        if not employees:
            return {
                'status': 'error',
                'message': 'No employees found for this payroll run'
            }
        
        # Update run status
        self.store.update_payroll_run(run_id, {'status': 'processing'})
        
        total_gross = 0
        total_net = 0
        total_taxes = 0
        
        # Process each employee
        for employee in employees:
            if employee.get('status') != 'active':
                continue
            
            payment = self._calculate_payment(employee, run)
            self.store.add_payment(payment)
            
            total_gross += payment['gross_pay']
            total_net += payment['net_pay']
            total_taxes += payment['federal_tax'] + payment['state_tax'] + payment['social_security'] + payment['medicare']
        
        # Update run with totals
        updates = {
            'status': 'completed',
            'total_gross_pay': total_gross,
            'total_net_pay': total_net,
            'total_taxes': total_taxes,
            'total_deductions': total_taxes,
            'employee_count': len(employees),
            'completed_at': datetime.now(timezone.utc).isoformat()
        }
        
        self.store.update_payroll_run(run_id, updates)
        
        run = self.store.get_payroll_run(run_id)
        
        self.logger.log_info(f"Payroll run processed: {run_id}", {'context': 'payroll_service'})
        
        return {
            'status': 'success',
            'run': run,
            'payments_processed': len(employees)
        }
    
    def _calculate_payment(self, employee: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate payment for a single employee"""
        
        payment_id = f"PAY-{secrets.token_hex(4).upper()}"
        
        # Calculate gross pay based on pay frequency
        pay_frequency = employee.get('pay_frequency', 'biweekly')
        annual_salary = employee.get('salary', 0)
        
        if pay_frequency == 'weekly':
            periods_per_year = 52
        elif pay_frequency == 'biweekly':
            periods_per_year = 26
        elif pay_frequency == 'monthly':
            periods_per_year = 12
        else:
            periods_per_year = 26
        
        gross_pay = annual_salary / periods_per_year
        
        # Calculate taxes
        annual_gross = annual_salary
        state_rate = employee.get('state_tax_rate', 0.05)
        
        taxes = self.tax_calculator.calculate_all_taxes(annual_gross, state_rate)
        
        # Pro-rate taxes for this pay period
        federal_tax = taxes['federal_tax'] / periods_per_year
        state_tax = taxes['state_tax'] / periods_per_year
        social_security = taxes['social_security'] / periods_per_year
        medicare = taxes['medicare'] / periods_per_year
        
        # Calculate other deductions
        health_insurance = employee.get('health_insurance_deduction', 0)
        retirement = employee.get('retirement_contribution', 0)
        
        # Calculate net pay
        total_deductions = federal_tax + state_tax + social_security + medicare + health_insurance + retirement
        net_pay = gross_pay - total_deductions
        
        payment = {
            'payment_id': payment_id,
            'employee_id': employee['employee_id'],
            'payroll_run_id': run['run_id'],
            'pay_period_start': run['pay_period_start'],
            'pay_period_end': run['pay_period_end'],
            'payment_date': run['payment_date'],
            'gross_pay': round(gross_pay, 2),
            'net_pay': round(net_pay, 2),
            'federal_tax': round(federal_tax, 2),
            'state_tax': round(state_tax, 2),
            'social_security': round(social_security, 2),
            'medicare': round(medicare, 2),
            'health_insurance': round(health_insurance, 2),
            'retirement_contribution': round(retirement, 2),
            'status': 'completed',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        return payment
    
    def get_payroll_run(self, run_id: str) -> Dict[str, Any]:
        """Get a payroll run by ID"""
        run = self.store.get_payroll_run(run_id)
        
        if not run:
            return {
                'status': 'error',
                'message': 'Payroll run not found'
            }
        
        # Get payments for this run
        payments = self.store.get_payments_by_run(run_id)
        
        return {
            'status': 'success',
            'run': run,
            'payments': payments
        }
    
    def list_payroll_runs(self, user_id: str) -> Dict[str, Any]:
        """List all payroll runs for a user"""
        runs = self.store.get_payroll_runs_by_user(user_id)
        
        return {
            'status': 'success',
            'runs': runs,
            'count': len(runs)
        }
    
    def get_payment(self, payment_id: str) -> Dict[str, Any]:
        """Get a payment by ID"""
        payment = self.store.get_payment(payment_id)
        
        if not payment:
            return {
                'status': 'error',
                'message': 'Payment not found'
            }
        
        return {
            'status': 'success',
            'payment': payment
        }
    
    def get_employee_payments(self, employee_id: str) -> Dict[str, Any]:
        """Get all payments for an employee"""
        payments = self.store.get_payments_by_employee(employee_id)
        
        return {
            'status': 'success',
            'payments': payments,
            'count': len(payments)
        }
    
    def calculate_employee_taxes(self, employee_id: str) -> Dict[str, Any]:
        """Calculate estimated taxes for an employee"""
        employee = self.store.get_employee(employee_id)
        
        if not employee:
            return {
                'status': 'error',
                'message': 'Employee not found'
            }
        
        annual_gross = employee.get('salary', 0)
        state_rate = employee.get('state_tax_rate', 0.05)
        
        taxes = self.tax_calculator.calculate_all_tasks(annual_gross, state_rate)
        
        return {
            'status': 'success',
            'employee_id': employee_id,
            'annual_gross': annual_gross,
            'taxes': taxes,
            'effective_tax_rate': sum(taxes.values()) / annual_gross if annual_gross > 0 else 0
        }


# Global service instance
payroll_service = PayrollService()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PayrollService',
    'payroll_service',
    'TaxCalculator',
    'InMemoryPayrollStore',
    'payroll_store'
]
