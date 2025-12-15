#!/usr/bin/env python3
"""
JPMorgan HR Benefits Management API
Handles employee benefits enrollment, management, and administration
"""
# pylint: disable=import-error,invalid-name,broad-exception-caught,line-too-long,unused-argument,reimported,ungrouped-imports,wrong-import-order,wrong-import-position,unspecified-encoding,missing-class-docstring,missing-function-docstring,superfluous-parens
import json
import os
import secrets
from datetime import datetime, timezone, timedelta
from functools import wraps

from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Audit Logging Imports
try:
    from src.audit_logger import AuditLogger, audit_log
    from src.database_fixed import db_manager
    from config import config
    AUDIT_LOGGING_AVAILABLE = True
except ImportError:
    AUDIT_LOGGING_AVAILABLE = False
    AuditLogger = None
    audit_log = None
    db_manager = None
    config = None

# Create Blueprint for HR Benefits
hr_bp = Blueprint('hr_benefits', __name__, url_prefix='/api/hr')

# Initialize Audit Logger
audit_logger = None
if AUDIT_LOGGING_AVAILABLE:
    try:
        audit_logger = AuditLogger(db_manager)
    except Exception as e:
        print(f"Failed to initialize audit logger: {e}")
        audit_logger = None

# In-memory storage for demo (replace with database in production)
employees = {}
benefits_plans = {}
enrollments = {}
claims = {}

# Initialize sample data
def init_sample_data():
    """Initialize sample HR data for demonstration"""

    # Sample benefits plans
    benefits_plans.update({
        'health_basic': {
            'plan_id': 'health_basic',
            'name': 'Basic Health Insurance',
            'type': 'health',
            'description': 'Basic medical, dental, and vision coverage',
            'monthly_cost': 150.00,
            'annual_deductible': 1500.00,
            'coverage_limit': 500000.00,
            'status': 'active'
        },
        'health_premium': {
            'plan_id': 'health_premium',
            'name': 'Premium Health Insurance',
            'type': 'health',
            'description': 'Comprehensive medical, dental, and vision coverage',
            'monthly_cost': 350.00,
            'annual_deductible': 500.00,
            'coverage_limit': 1000000.00,
            'status': 'active'
        },
        'dental_basic': {
            'plan_id': 'dental_basic',
            'name': 'Basic Dental Coverage',
            'type': 'dental',
            'description': 'Preventive and basic dental care',
            'monthly_cost': 25.00,
            'annual_limit': 1500.00,
            'status': 'active'
        },
        'vision_basic': {
            'plan_id': 'vision_basic',
            'name': 'Basic Vision Coverage',
            'type': 'vision',
            'description': 'Eye exams and basic eyewear',
            'monthly_cost': 15.00,
            'annual_limit': 200.00,
            'status': 'active'
        },
        'life_term': {
            'plan_id': 'life_term',
            'name': 'Term Life Insurance',
            'type': 'life',
            'description': 'Term life insurance coverage',
            'monthly_cost': 20.00,
            'coverage_amount': 250000.00,
            'term_years': 20,
            'status': 'active'
        },
        'retirement_401k': {
            'plan_id': 'retirement_401k',
            'name': '401(k) Retirement Plan',
            'type': 'retirement',
            'description': 'Tax-advantaged retirement savings',
            'employee_contribution_limit': 19500.00,
            'employer_match_percent': 0.05,
            'status': 'active'
        }
    })

    # Sample employees
    employees.update({
        'EMP001': {
            'employee_id': 'EMP001',
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john.doe@jpmorgan.com',
            'department': 'Investment Banking',
            'hire_date': '2020-01-15',
            'salary': 150000.00,
            'status': 'active'
        },
        'EMP002': {
            'employee_id': 'EMP002',
            'first_name': 'Jane',
            'last_name': 'Smith',
            'email': 'jane.smith@jpmorgan.com',
            'department': 'Risk Management',
            'hire_date': '2019-03-20',
            'salary': 135000.00,
            'status': 'active'
        },
        'EMP003': {
            'employee_id': 'EMP003',
            'first_name': 'Michael',
            'last_name': 'Johnson',
            'email': 'michael.johnson@jpmorgan.com',
            'department': 'Technology',
            'hire_date': '2021-06-10',
            'salary': 120000.00,
            'status': 'active'
        }
    })

# Initialize sample data on module load
init_sample_data()

def hr_token_required(f):
    """Decorator to require HR authentication token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401

        token = auth_header.split(' ')[1]
        # For demo, accept any token that starts with 'hr_'
        if not token.startswith('hr_'):
            return jsonify({'error': 'Invalid HR token'}), 401

        return f(*args, **kwargs)
    return decorated_function

# Employee Management Endpoints
@hr_bp.route('/employees', methods=['GET'])
@hr_token_required
@audit_log(action='get_employees', resource_type='employee', category='hr_management') if AUDIT_LOGGING_AVAILABLE and audit_log else lambda f: f
def get_employees():
    """Get all employees"""
    try:
        employee_list = list(employees.values())
        return jsonify({
            'status': 'success',
            'employees': employee_list,
            'count': len(employee_list),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@hr_bp.route('/employees/<employee_id>', methods=['GET'])
@hr_token_required
@audit_log(action='get_employee', resource_type='employee', category='hr_management') if AUDIT_LOGGING_AVAILABLE and audit_log else lambda f: f
def get_employee(employee_id):
    """Get employee details"""
    try:
        employee = employees.get(employee_id)
        if not employee:
            return jsonify({'error': 'Employee not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'employee': employee,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Benefits Management Endpoints
@hr_bp.route('/benefits/plans', methods=['GET'])
@hr_token_required
def get_benefits_plans():
    """Get all benefits plans"""
    try:
        plans_list = list(benefits_plans.values())
        return jsonify({
            'status': 'success',
            'plans': plans_list,
            'count': len(plans_list),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@hr_bp.route('/benefits/plans/<plan_id>', methods=['GET'])
@hr_token_required
@audit_log(action='get_benefits_plan', resource_type='benefits_plan', category='hr_management') if AUDIT_LOGGING_AVAILABLE and audit_log else lambda f: f
def get_benefits_plan(plan_id):
    """Get specific benefits plan"""
    try:
        plan = benefits_plans.get(plan_id)
        if not plan:
            return jsonify({'error': 'Benefits plan not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'plan': plan,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Enrollment Management Endpoints
@hr_bp.route('/benefits/enrollments', methods=['GET'])
    """Get all benefits enrollments"""
@audit_log(action='get_enrollments', resource_type='enrollment', category='hr_management') if AUDIT_LOGGING_AVAILABLE and audit_log else lambda f: f
def get_enrollments():
    try:
        enrollment_list = list(enrollments.values())
        return jsonify({
            'status': 'success',
            'enrollments': enrollment_list,
            'count': len(enrollment_list),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@hr_bp.route('/benefits/enrollments', methods=['POST'])
@hr_token_required
def create_enrollment():
    """Create new benefits enrollment"""
    try:
        data = request.get_json(force=True)
        employee_id = data.get('employee_id')
        plan_id = data.get('plan_id')

        if not employee_id or not plan_id:
            return jsonify({'error': 'Employee ID and Plan ID are required', 'status': 'error'}), 400

        if employee_id not in employees:
            return jsonify({'error': 'Employee not found', 'status': 'error'}), 404

        if plan_id not in benefits_plans:
            return jsonify({'error': 'Benefits plan not found', 'status': 'error'}), 404

        enrollment_id = f"ENR{secrets.token_hex(4).upper()}"
        enrollment = {
            'enrollment_id': enrollment_id,
            'employee_id': employee_id,
            'plan_id': plan_id,
            'enrollment_date': datetime.now(timezone.utc).date().isoformat(),
            'status': 'active',
            'monthly_contribution': benefits_plans[plan_id]['monthly_cost'],
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        enrollments[enrollment_id] = enrollment

        return jsonify({
            'status': 'success',
            'enrollment': enrollment,
            'message': 'Benefits enrollment created successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@hr_bp.route('/benefits/enrollments/<enrollment_id>', methods=['GET'])
@hr_token_required
def get_enrollment(enrollment_id):
    """Get enrollment details"""
    try:
        enrollment = enrollments.get(enrollment_id)
        if not enrollment:
            return jsonify({'error': 'Enrollment not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'enrollment': enrollment,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@hr_bp.route('/employees/<employee_id>/benefits', methods=['GET'])
@hr_token_required
def get_employee_benefits(employee_id):
    """Get benefits for specific employee"""
    try:
        if employee_id not in employees:
            return jsonify({'error': 'Employee not found', 'status': 'error'}), 404

        employee_enrollments = [
            enrollment for enrollment in enrollments.values()
            if enrollment['employee_id'] == employee_id and enrollment['status'] == 'active'
        ]

        # Get plan details for each enrollment
        benefits_details = []
        for enrollment in employee_enrollments:
            plan = benefits_plans.get(enrollment['plan_id'])
            if plan:
                benefits_details.append({
                    'enrollment_id': enrollment['enrollment_id'],
                    'plan': plan,
                    'enrollment_date': enrollment['enrollment_date'],
                    'monthly_contribution': enrollment['monthly_contribution'],
                    'status': enrollment['status']
                })

        return jsonify({
            'status': 'success',
            'employee_id': employee_id,
            'benefits': benefits_details,
            'count': len(benefits_details),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Claims Management Endpoints
@hr_bp.route('/benefits/claims', methods=['GET'])
@hr_token_required
def get_claims():
    """Get all benefits claims"""
    try:
        claims_list = list(claims.values())
        return jsonify({
            'status': 'success',
            'claims': claims_list,
            'count': len(claims_list),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@hr_bp.route('/benefits/claims', methods=['POST'])
@hr_token_required
def submit_claim():
    """Submit new benefits claim"""
    try:
        data = request.get_json(force=True)
        employee_id = data.get('employee_id')
        claim_type = data.get('claim_type')
        amount = data.get('amount')
        description = data.get('description', '')

        if not all([employee_id, claim_type, amount]):
            return jsonify({'error': 'Employee ID, claim type, and amount are required', 'status': 'error'}), 400

        if employee_id not in employees:
            return jsonify({'error': 'Employee not found', 'status': 'error'}), 404

        claim_id = f"CLM{secrets.token_hex(4).upper()}"
        claim = {
            'claim_id': claim_id,
            'employee_id': employee_id,
            'claim_type': claim_type,
            'amount': float(amount),
            'description': description,
            'status': 'pending',
            'submitted_date': datetime.now(timezone.utc).date().isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        claims[claim_id] = claim

        return jsonify({
            'status': 'success',
            'claim': claim,
            'message': 'Benefits claim submitted successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@hr_bp.route('/benefits/claims/<claim_id>/approve', methods=['POST'])
@hr_token_required
def approve_claim(claim_id):
    """Approve benefits claim"""
    try:
        claim = claims.get(claim_id)
        if not claim:
            return jsonify({'error': 'Claim not found', 'status': 'error'}), 404

        if claim['status'] != 'pending':
            return jsonify({'error': 'Claim is not in pending status', 'status': 'error'}), 400

        claim['status'] = 'approved'
        claim['approved_date'] = datetime.now(timezone.utc).date().isoformat()
        claim['updated_at'] = datetime.now(timezone.utc).isoformat()

        return jsonify({
            'status': 'success',
            'claim': claim,
            'message': 'Claim approved successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@hr_bp.route('/benefits/claims/<claim_id>/deny', methods=['POST'])
@hr_token_required
def deny_claim(claim_id):
    """Deny benefits claim"""
    try:
        data = request.get_json(force=True)
        denial_reason = data.get('reason', 'No reason provided')

        claim = claims.get(claim_id)
        if not claim:
            return jsonify({'error': 'Claim not found', 'status': 'error'}), 404

        if claim['status'] != 'pending':
            return jsonify({'error': 'Claim is not in pending status', 'status': 'error'}), 400

        claim['status'] = 'denied'
        claim['denial_reason'] = denial_reason
        claim['denied_date'] = datetime.now(timezone.utc).date().isoformat()
        claim['updated_at'] = datetime.now(timezone.utc).isoformat()

        return jsonify({
            'status': 'success',
            'claim': claim,
            'message': 'Claim denied successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Payroll Integration Endpoints
@hr_bp.route('/payroll/salary', methods=['GET'])
@hr_token_required
def get_salary_info():
    """Get employee salary information"""
    try:
        employee_id = request.args.get('employee_id')
        if not employee_id:
            return jsonify({'error': 'Employee ID is required', 'status': 'error'}), 400

        employee = employees.get(employee_id)
        if not employee:
            return jsonify({'error': 'Employee not found', 'status': 'error'}), 404

        salary_info = {
            'employee_id': employee_id,
            'annual_salary': employee['salary'],
            'monthly_salary': employee['salary'] / 12,
            'biweekly_salary': employee['salary'] / 26,
            'hourly_rate': employee['salary'] / 2080,  # Assuming 40 hours/week * 52 weeks
            'pay_period': 'biweekly',
            'last_pay_date': (datetime.now(timezone.utc) - timedelta(days=14)).date().isoformat(),
            'next_pay_date': (datetime.now(timezone.utc) + timedelta(days=14)).date().isoformat()
        }

        return jsonify({
            'status': 'success',
            'salary_info': salary_info,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@hr_bp.route('/payroll/deductions', methods=['GET'])
@hr_token_required
def get_payroll_deductions():
    """Get payroll deductions for employee"""
    try:
        employee_id = request.args.get('employee_id')
        if not employee_id:
            return jsonify({'error': 'Employee ID is required', 'status': 'error'}), 400

        employee = employees.get(employee_id)
        if not employee:
            return jsonify({'error': 'Employee not found', 'status': 'error'}), 404

        # Calculate deductions based on salary
        annual_salary = employee['salary']
        deductions = {
            'federal_tax': annual_salary * 0.22,  # 22% federal tax
            'social_security': min(annual_salary * 0.062, 160200 * 0.062),  # 6.2% up to wage base
            'medicare': annual_salary * 0.0145,  # 1.45% medicare
            'state_tax': annual_salary * 0.05,  # 5% state tax (example)
            'health_insurance': 150.00 * 12,  # Monthly premium * 12
            'dental_insurance': 25.00 * 12,
            'vision_insurance': 15.00 * 12,
            'retirement_401k': annual_salary * 0.06  # 6% employee contribution
        }

        total_deductions = sum(deductions.values())
        net_annual = annual_salary - total_deductions

        return jsonify({
            'status': 'success',
            'employee_id': employee_id,
            'gross_annual': annual_salary,
            'deductions': deductions,
            'total_deductions': total_deductions,
            'net_annual': net_annual,
            'net_monthly': net_annual / 12,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# HR Analytics Endpoints
@hr_bp.route('/analytics/benefits', methods=['GET'])
@hr_token_required
def get_benefits_analytics():
    """Get benefits enrollment analytics"""
    try:
        total_employees = len(employees)
        total_enrollments = len([e for e in enrollments.values() if e['status'] == 'active'])
        total_claims = len(claims)

        # Calculate enrollment rates by plan type
        plan_enrollments = {}
        for enrollment in enrollments.values():
            if enrollment['status'] == 'active':
                plan_id = enrollment['plan_id']
                plan_type = benefits_plans.get(plan_id, {}).get('type', 'unknown')
                plan_enrollments[plan_type] = plan_enrollments.get(plan_type, 0) + 1

        analytics = {
            'total_employees': total_employees,
            'total_active_enrollments': total_enrollments,
            'enrollment_rate': (total_enrollments / total_employees) * 100 if total_employees > 0 else 0,
            'total_claims': total_claims,
            'enrollments_by_type': plan_enrollments,
            'average_monthly_contribution': sum(e['monthly_contribution'] for e in enrollments.values() if e['status'] == 'active') / total_enrollments if total_enrollments > 0 else 0
        }

        return jsonify({
            'status': 'success',
            'analytics': analytics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@hr_bp.route('/analytics/payroll', methods=['GET'])
@hr_token_required
def get_payroll_analytics():
    """Get payroll analytics"""
    try:
        total_employees = len(employees)
        total_salary = sum(emp['salary'] for emp in employees.values())

        analytics = {
            'total_employees': total_employees,
            'total_annual_payroll': total_salary,
            'average_salary': total_salary / total_employees if total_employees > 0 else 0,
            'total_monthly_payroll': total_salary / 12,
            'total_benefits_cost': sum(
                sum(e['monthly_contribution'] for e in enrollments.values() if e['status'] == 'active' and e['employee_id'] == emp_id)
                for emp_id in employees.keys()
            ) * 12
        }

        return jsonify({
            'status': 'success',
            'analytics': analytics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Export functions for integration
def get_hr_blueprint():
    """Get the HR benefits blueprint for integration"""
    return hr_bp

def get_hr_endpoints():
    """Get list of HR endpoints for documentation"""
    return [
        '/api/hr/employees - Employee management',
        '/api/hr/benefits/plans - Benefits plans management',
        '/api/hr/benefits/enrollments - Benefits enrollments',
        '/api/hr/benefits/claims - Benefits claims management',
        '/api/hr/payroll/salary - Salary information',
        '/api/hr/payroll/deductions - Payroll deductions',
        '/api/hr/analytics/benefits - Benefits analytics',
        '/api/hr/analytics/payroll - Payroll analytics'
    ]
