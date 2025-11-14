#!/usr/bin/env python3
"""
JPMorgan Payroll Processing API
Handles payroll calculations, disbursements, and payroll management
"""
# pylint: disable=import-error,invalid-name,broad-exception-caught,line-too-long,unused-argument,reimported,ungrouped-imports,wrong-import-order,wrong-import-position,unspecified-encoding,missing-class-docstring,missing-function-docstring,superfluous-parens
import json
import os
import secrets
from datetime import datetime, timezone, timedelta
from functools import wraps

from flask import Blueprint, request, jsonify

# Create Blueprint for Payroll
payroll_bp = Blueprint('payroll', __name__, url_prefix='/api/payroll')

# In-memory storage for demo (replace with database in production)
payroll_records = {}
payroll_schedules = {}
tax_configurations = {}
direct_deposit_accounts = {}

# Initialize sample data
def init_payroll_data():
    """Initialize sample payroll data for demonstration"""

    # Tax configurations (simplified for demo)
    tax_configurations.update({
        'federal': {
            'type': 'federal_income',
            'brackets': [
                {'min': 0, 'max': 11000, 'rate': 0.10},
                {'min': 11000, 'max': 44725, 'rate': 0.12},
                {'min': 44725, 'max': 95375, 'rate': 0.22},
                {'min': 95375, 'max': 182100, 'rate': 0.24},
                {'min': 182100, 'max': 231250, 'rate': 0.32},
                {'min': 231250, 'max': 578125, 'rate': 0.35},
                {'min': 578125, 'max': float('inf'), 'rate': 0.37}
            ],
            'standard_deduction': 13850,  # Single filer
            'pay_periods': 26  # Biweekly
        },
        'social_security': {
            'type': 'social_security',
            'rate': 0.062,
            'wage_base': 160200,
            'max_amount': 9938.40  # 160200 * 0.062
        },
        'medicare': {
            'type': 'medicare',
            'rate': 0.0145,
            'additional_rate': 0.009,  # Additional Medicare tax for high earners
            'additional_threshold': 200000
        },
        'state_ny': {
            'type': 'state_income',
            'state': 'NY',
            'brackets': [
                {'min': 0, 'max': 8500, 'rate': 0.04},
                {'min': 8500, 'max': 117000, 'rate': 0.045},
                {'min': 117000, 'max': 139000, 'rate': 0.0525},
                {'min': 139000, 'max': 214000, 'rate': 0.055},
                {'min': 214000, 'max': 235000, 'rate': 0.06},
                {'min': 235000, 'max': 1077550, 'rate': 0.0685},
                {'min': 1077550, 'max': 1575000, 'rate': 0.0965},
                {'min': 1575000, 'max': 2118750, 'rate': 0.103},
                {'min': 2118750, 'max': 2657180, 'rate': 0.109},
                {'min': 2657180, 'max': float('inf'), 'rate': 0.113}
            ],
            'standard_deduction': 8000
        }
    })

    # Payroll schedules
    payroll_schedules.update({
        'biweekly': {
            'schedule_id': 'biweekly',
            'name': 'Biweekly Payroll',
            'frequency': 'biweekly',
            'pay_periods_per_year': 26,
            'next_pay_date': (datetime.now(timezone.utc) + timedelta(days=14)).date().isoformat(),
            'status': 'active'
        },
        'semimonthly': {
            'schedule_id': 'semimonthly',
            'name': 'Semi-Monthly Payroll',
            'frequency': 'semimonthly',
            'pay_periods_per_year': 24,
            'next_pay_date': (datetime.now(timezone.utc) + timedelta(days=15)).date().isoformat(),
            'status': 'active'
        },
        'monthly': {
            'schedule_id': 'monthly',
            'name': 'Monthly Payroll',
            'frequency': 'monthly',
            'pay_periods_per_year': 12,
            'next_pay_date': (datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat(),
            'status': 'active'
        }
    })

    # Sample direct deposit accounts
    direct_deposit_accounts.update({
        'EMP001': {
            'employee_id': 'EMP001',
            'bank_name': 'Chase Bank',
            'account_number': '****1234',
            'routing_number': '021000021',
            'account_type': 'checking',
            'is_primary': True
        },
        'EMP002': {
            'employee_id': 'EMP002',
            'bank_name': 'Bank of America',
            'account_number': '****5678',
            'routing_number': '121000358',
            'account_type': 'checking',
            'is_primary': True
        }
    })

# Initialize sample data on module load
init_payroll_data()

def payroll_token_required(f):
    """Decorator to require payroll authentication token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401

        token = auth_header.split(' ')[1]
        # For demo, accept any token that starts with 'payroll_'
        if not token.startswith('payroll_'):
            return jsonify({'error': 'Invalid payroll token'}), 401

        return f(*args, **kwargs)
    return decorated_function

# Payroll Calculation Functions
def calculate_federal_tax(annual_salary, pay_periods=26):
    """Calculate federal income tax withholding"""
    taxable_income = annual_salary - tax_configurations['federal']['standard_deduction']
    if taxable_income <= 0:
        return 0

    tax = 0
    for bracket in tax_configurations['federal']['brackets']:
        if taxable_income > bracket['min']:
            taxable_in_bracket = min(taxable_income - bracket['min'], bracket['max'] - bracket['min'])
            if bracket['max'] == float('inf'):
                taxable_in_bracket = taxable_income - bracket['min']
            tax += taxable_in_bracket * bracket['rate']

    return tax / pay_periods

def calculate_social_security(annual_salary, pay_periods=26, ytd_earnings=0):
    """Calculate Social Security tax"""
    remaining_wage_base = max(0, tax_configurations['social_security']['wage_base'] - ytd_earnings)
    taxable_amount = min(annual_salary / pay_periods, remaining_wage_base)
    return taxable_amount * tax_configurations['social_security']['rate']

def calculate_medicare(annual_salary, pay_periods=26, ytd_earnings=0):
    """Calculate Medicare tax"""
    period_gross = annual_salary / pay_periods
    medicare_tax = period_gross * tax_configurations['medicare']['rate']

    # Additional Medicare tax for high earners
    ytd_plus_current = ytd_earnings + period_gross
    if ytd_plus_current > tax_configurations['medicare']['additional_threshold']:
        additional_taxable = max(0, period_gross - (tax_configurations['medicare']['additional_threshold'] - ytd_earnings))
        medicare_tax += additional_taxable * tax_configurations['medicare']['additional_rate']

    return medicare_tax

def calculate_state_tax(annual_salary, state='NY', pay_periods=26):
    """Calculate state income tax"""
    state_config = tax_configurations.get(f'state_{state.lower()}')
    if not state_config:
        return 0

    taxable_income = annual_salary - state_config['standard_deduction']
    if taxable_income <= 0:
        return 0

    tax = 0
    for bracket in state_config['brackets']:
        if taxable_income > bracket['min']:
            taxable_in_bracket = min(taxable_income - bracket['min'], bracket['max'] - bracket['min'])
            if bracket['max'] == float('inf'):
                taxable_in_bracket = taxable_income - bracket['min']
            tax += taxable_in_bracket * bracket['rate']

    return tax / pay_periods

# Payroll Processing Endpoints
@payroll_bp.route('/calculate', methods=['POST'])
@payroll_token_required
def calculate_payroll():
    """Calculate payroll for an employee"""
    try:
        data = request.get_json(force=True)
        employee_id = data.get('employee_id')
        annual_salary = data.get('annual_salary')
        pay_period = data.get('pay_period', 'biweekly')
        state = data.get('state', 'NY')
        ytd_earnings = data.get('ytd_earnings', 0)

        if not employee_id or not annual_salary:
            return jsonify({'error': 'Employee ID and annual salary are required', 'status': 'error'}), 400

        # Determine pay periods
        pay_periods = {
            'weekly': 52,
            'biweekly': 26,
            'semimonthly': 24,
            'monthly': 12
        }.get(pay_period, 26)

        period_gross = annual_salary / pay_periods

        # Calculate taxes
        federal_tax = calculate_federal_tax(annual_salary, pay_periods)
        social_security = calculate_social_security(annual_salary, pay_periods, ytd_earnings)
        medicare = calculate_medicare(annual_salary, pay_periods, ytd_earnings)
        state_tax = calculate_state_tax(annual_salary, state, pay_periods)

        # Calculate deductions (benefits, retirement, etc.)
        benefits_deductions = data.get('benefits_deductions', 0)
        retirement_contribution = data.get('retirement_contribution', 0)

        total_taxes = federal_tax + social_security + medicare + state_tax
        total_deductions = total_taxes + benefits_deductions + retirement_contribution
        net_pay = period_gross - total_deductions

        calculation = {
            'employee_id': employee_id,
            'pay_period': pay_period,
            'period_gross': round(period_gross, 2),
            'taxes': {
                'federal_income': round(federal_tax, 2),
                'social_security': round(social_security, 2),
                'medicare': round(medicare, 2),
                'state_income': round(state_tax, 2),
                'total_taxes': round(total_taxes, 2)
            },
            'deductions': {
                'benefits': round(benefits_deductions, 2),
                'retirement': round(retirement_contribution, 2),
                'total_deductions': round(total_deductions, 2)
            },
            'net_pay': round(net_pay, 2),
            'annualized_net': round(net_pay * pay_periods, 2)
        }

        return jsonify({
            'status': 'success',
            'calculation': calculation,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@payroll_bp.route('/run', methods=['POST'])
@payroll_token_required
def run_payroll():
    """Run payroll for multiple employees"""
    try:
        data = request.get_json(force=True)
        employees = data.get('employees', [])
        pay_date = data.get('pay_date', datetime.now(timezone.utc).date().isoformat())
        pay_period = data.get('pay_period', 'biweekly')

        if not employees:
            return jsonify({'error': 'Employee list is required', 'status': 'error'}), 400

        payroll_results = []
        total_gross = 0
        total_net = 0
        total_taxes = 0

        for emp_data in employees:
            # Calculate payroll for each employee
            calc_data = {
                'employee_id': emp_data['employee_id'],
                'annual_salary': emp_data['annual_salary'],
                'pay_period': pay_period,
                'state': emp_data.get('state', 'NY'),
                'ytd_earnings': emp_data.get('ytd_earnings', 0),
                'benefits_deductions': emp_data.get('benefits_deductions', 0),
                'retirement_contribution': emp_data.get('retirement_contribution', 0)
            }

            # Use the calculate endpoint logic
            response = calculate_payroll()
            if response[1] == 200:
                calc_result = response[0].get_json()['calculation']
                payroll_results.append(calc_result)

                total_gross += calc_result['period_gross']
                total_net += calc_result['net_pay']
                total_taxes += calc_result['taxes']['total_taxes']
            else:
                payroll_results.append({
                    'employee_id': emp_data['employee_id'],
                    'error': 'Calculation failed'
                })

        # Create payroll record
        payroll_id = f"PR{secrets.token_hex(4).upper()}"
        payroll_record = {
            'payroll_id': payroll_id,
            'pay_date': pay_date,
            'pay_period': pay_period,
            'employee_count': len([r for r in payroll_results if 'error' not in r]),
            'total_gross': round(total_gross, 2),
            'total_net': round(total_net, 2),
            'total_taxes': round(total_taxes, 2),
            'status': 'processed',
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        payroll_records[payroll_id] = payroll_record

        return jsonify({
            'status': 'success',
            'payroll_record': payroll_record,
            'employee_payroll': payroll_results,
            'summary': {
                'total_employees': len(payroll_results),
                'successful_calculations': len([r for r in payroll_results if 'error' not in r]),
                'total_gross_payroll': round(total_gross, 2),
                'total_net_payroll': round(total_net, 2),
                'total_tax_withheld': round(total_taxes, 2)
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@payroll_bp.route('/records', methods=['GET'])
@payroll_token_required
def get_payroll_records():
    """Get payroll records"""
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        records = list(payroll_records.values())
        records.sort(key=lambda x: x['created_at'], reverse=True)

        paginated_records = records[offset:offset + limit]

        return jsonify({
            'status': 'success',
            'records': paginated_records,
            'total_count': len(records),
            'limit': limit,
            'offset': offset,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@payroll_bp.route('/records/<payroll_id>', methods=['GET'])
@payroll_token_required
def get_payroll_record(payroll_id):
    """Get specific payroll record"""
    try:
        record = payroll_records.get(payroll_id)
        if not record:
            return jsonify({'error': 'Payroll record not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'record': record,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Direct Deposit Management
@payroll_bp.route('/direct-deposit', methods=['GET'])
@payroll_token_required
def get_direct_deposit_accounts():
    """Get direct deposit accounts"""
    try:
        accounts = list(direct_deposit_accounts.values())
        return jsonify({
            'status': 'success',
            'accounts': accounts,
            'count': len(accounts),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@payroll_bp.route('/direct-deposit', methods=['POST'])
@payroll_token_required
def setup_direct_deposit():
    """Setup direct deposit for employee"""
    try:
        data = request.get_json(force=True)
        employee_id = data.get('employee_id')
        bank_name = data.get('bank_name')
        account_number = data.get('account_number')
        routing_number = data.get('routing_number')
        account_type = data.get('account_type', 'checking')

        if not all([employee_id, bank_name, account_number, routing_number]):
            return jsonify({'error': 'All bank account details are required', 'status': 'error'}), 400

        # Mask account number for security
        masked_account = f"****{account_number[-4:]}"

        account = {
            'employee_id': employee_id,
            'bank_name': bank_name,
            'account_number': masked_account,
            'routing_number': routing_number,
            'account_type': account_type,
            'is_primary': len([a for a in direct_deposit_accounts.values() if a['employee_id'] == employee_id]) == 0,
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        # Use employee_id as key (assuming one account per employee for demo)
        direct_deposit_accounts[employee_id] = account

        return jsonify({
            'status': 'success',
            'account': account,
            'message': 'Direct deposit account setup successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Tax Configuration Endpoints
@payroll_bp.route('/tax-config', methods=['GET'])
@payroll_token_required
def get_tax_config():
    """Get tax configuration"""
    try:
        return jsonify({
            'status': 'success',
            'tax_config': tax_configurations,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@payroll_bp.route('/tax-config/<config_type>', methods=['PUT'])
@payroll_token_required
def update_tax_config(config_type):
    """Update tax configuration"""
    try:
        data = request.get_json(force=True)

        if config_type not in tax_configurations:
            return jsonify({'error': 'Tax configuration type not found', 'status': 'error'}), 404

        # Update configuration (in production, validate thoroughly)
        tax_configurations[config_type].update(data)

        return jsonify({
            'status': 'success',
            'config_type': config_type,
            'updated_config': tax_configurations[config_type],
            'message': 'Tax configuration updated successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Payroll Analytics
@payroll_bp.route('/analytics', methods=['GET'])
@payroll_token_required
def get_payroll_analytics():
    """Get payroll analytics"""
    try:
        records = list(payroll_records.values())

        if not records:
            return jsonify({
                'status': 'success',
                'analytics': {
                    'total_payrolls': 0,
                    'total_employees_paid': 0,
                    'total_gross_payroll': 0,
                    'total_net_payroll': 0,
                    'average_payroll': 0
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200

        total_payrolls = len(records)
        total_gross = sum(r['total_gross'] for r in records)
        total_net = sum(r['total_net'] for r in records)
        total_employees = sum(r['employee_count'] for r in records)

        analytics = {
            'total_payrolls': total_payrolls,
            'total_employees_paid': total_employees,
            'total_gross_payroll': round(total_gross, 2),
            'total_net_payroll': round(total_net, 2),
            'average_payroll_gross': round(total_gross / total_payrolls, 2) if total_payrolls > 0 else 0,
            'average_payroll_net': round(total_net / total_payrolls, 2) if total_payrolls > 0 else 0,
            'average_employees_per_payroll': round(total_employees / total_payrolls, 2) if total_payrolls > 0 else 0
        }

        return jsonify({
            'status': 'success',
            'analytics': analytics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Export functions for integration
def get_payroll_blueprint():
    """Get the payroll blueprint for integration"""
    return payroll_bp

def get_payroll_endpoints():
    """Get list of payroll endpoints for documentation"""
    return [
        '/api/payroll/calculate - Calculate payroll for employee',
        '/api/payroll/run - Run payroll for multiple employees',
        '/api/payroll/records - Payroll records management',
        '/api/payroll/direct-deposit - Direct deposit account management',
        '/api/payroll/tax-config - Tax configuration management',
        '/api/payroll/analytics - Payroll analytics and reporting'
    ]
