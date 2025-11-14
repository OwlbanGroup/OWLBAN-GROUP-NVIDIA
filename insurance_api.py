#!/usr/bin/env python3
"""
JPMorgan Insurance Management API
Handles insurance claims, coverage management, and insurance administration
"""
# pylint: disable=import-error,invalid-name,broad-exception-caught,line-too-long,unused-argument,reimported,ungrouped-imports,wrong-import-order,wrong-import-position,unspecified-encoding,missing-class-docstring,missing-function-docstring,superfluous-parens
import json
import os
import secrets
from datetime import datetime, timezone, timedelta
from functools import wraps

from flask import Blueprint, request, jsonify

# Create Blueprint for Insurance
insurance_bp = Blueprint('insurance', __name__, url_prefix='/api/insurance')

# In-memory storage for demo (replace with database in production)
insurance_policies = {}
insurance_claims = {}
coverage_types = {}
underwriting_rules = {}

# Initialize sample data
def init_insurance_data():
    """Initialize sample insurance data for demonstration"""

    # Coverage types
    coverage_types.update({
        'health_basic': {
            'coverage_id': 'health_basic',
            'name': 'Basic Health Coverage',
            'type': 'health',
            'description': 'Essential medical coverage',
            'annual_limit': 500000.00,
            'deductible': 1500.00,
            'coinsurance': 0.20,
            'monthly_premium': 150.00,
            'status': 'active'
        },
        'health_comprehensive': {
            'coverage_id': 'health_comprehensive',
            'name': 'Comprehensive Health Coverage',
            'type': 'health',
            'description': 'Full medical, dental, and vision coverage',
            'annual_limit': 1000000.00,
            'deductible': 500.00,
            'coinsurance': 0.10,
            'monthly_premium': 350.00,
            'status': 'active'
        },
        'dental_preventive': {
            'coverage_id': 'dental_preventive',
            'name': 'Preventive Dental Coverage',
            'type': 'dental',
            'description': 'Cleanings, exams, and preventive care',
            'annual_limit': 1500.00,
            'deductible': 0.00,
            'coinsurance': 0.00,
            'monthly_premium': 25.00,
            'status': 'active'
        },
        'vision_exams': {
            'coverage_id': 'vision_exams',
            'name': 'Vision Exam Coverage',
            'type': 'vision',
            'description': 'Annual eye exams and basic eyewear',
            'annual_limit': 200.00,
            'deductible': 0.00,
            'coinsurance': 0.00,
            'monthly_premium': 15.00,
            'status': 'active'
        },
        'life_term': {
            'coverage_id': 'life_term',
            'name': 'Term Life Insurance',
            'type': 'life',
            'description': 'Term life insurance coverage',
            'coverage_amount': 250000.00,
            'term_years': 20,
            'monthly_premium': 20.00,
            'status': 'active'
        },
        'disability_short': {
            'coverage_id': 'disability_short',
            'name': 'Short-Term Disability',
            'type': 'disability',
            'description': 'Short-term disability income protection',
            'monthly_benefit': 2000.00,
            'elimination_period': 7,  # days
            'benefit_period': 26,  # weeks
            'monthly_premium': 45.00,
            'status': 'active'
        },
        'disability_long': {
            'coverage_id': 'disability_long',
            'name': 'Long-Term Disability',
            'type': 'disability',
            'description': 'Long-term disability income protection',
            'monthly_benefit_percentage': 0.60,  # 60% of salary
            'elimination_period': 90,  # days
            'benefit_period': 60,  # months
            'monthly_premium': 85.00,
            'status': 'active'
        }
    })

    # Underwriting rules
    underwriting_rules.update({
        'health_standard': {
            'rule_id': 'health_standard',
            'name': 'Standard Health Underwriting',
            'coverage_type': 'health',
            'age_brackets': [
                {'min_age': 18, 'max_age': 35, 'multiplier': 1.0},
                {'min_age': 36, 'max_age': 45, 'multiplier': 1.2},
                {'min_age': 46, 'max_age': 55, 'multiplier': 1.5},
                {'min_age': 56, 'max_age': 65, 'multiplier': 2.0}
            ],
            'health_factors': {
                'smoker': 1.5,
                'obese': 1.3,
                'chronic_condition': 1.8,
                'excellent_health': 0.8
            }
        },
        'life_standard': {
            'rule_id': 'life_standard',
            'name': 'Standard Life Underwriting',
            'coverage_type': 'life',
            'age_brackets': [
                {'min_age': 18, 'max_age': 30, 'multiplier': 0.5},
                {'min_age': 31, 'max_age': 40, 'multiplier': 0.8},
                {'min_age': 41, 'max_age': 50, 'multiplier': 1.2},
                {'min_age': 51, 'max_age': 60, 'multiplier': 2.5},
                {'min_age': 61, 'max_age': 70, 'multiplier': 4.0}
            ],
            'risk_factors': {
                'smoker': 3.0,
                'dangerous_hobby': 1.5,
                'family_history': 1.3,
                'excellent_health': 0.7
            }
        }
    })

# Initialize sample data on module load
init_insurance_data()

def insurance_token_required(f):
    """Decorator to require insurance authentication token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401

        token = auth_header.split(' ')[1]
        # For demo, accept any token that starts with 'insurance_'
        if not token.startswith('insurance_'):
            return jsonify({'error': 'Invalid insurance token'}), 401

        return f(*args, **kwargs)
    return decorated_function

# Policy Management Endpoints
@insurance_bp.route('/policies', methods=['GET'])
@insurance_token_required
def get_policies():
    """Get all insurance policies"""
    try:
        policies = list(insurance_policies.values())
        return jsonify({
            'status': 'success',
            'policies': policies,
            'count': len(policies),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@insurance_bp.route('/policies', methods=['POST'])
@insurance_token_required
def create_policy():
    """Create new insurance policy"""
    try:
        data = request.get_json(force=True)
        employee_id = data.get('employee_id')
        coverage_id = data.get('coverage_id')
        effective_date = data.get('effective_date', datetime.now(timezone.utc).date().isoformat())

        if not employee_id or not coverage_id:
            return jsonify({'error': 'Employee ID and Coverage ID are required', 'status': 'error'}), 400

        if coverage_id not in coverage_types:
            return jsonify({'error': 'Invalid coverage type', 'status': 'error'}), 404

        coverage = coverage_types[coverage_id]

        # Calculate premium based on underwriting rules
        base_premium = coverage['monthly_premium']
        # For demo, apply simple age-based adjustment
        age = data.get('age', 35)
        age_multiplier = 1.0
        if age > 50:
            age_multiplier = 1.5
        elif age > 40:
            age_multiplier = 1.2

        final_premium = base_premium * age_multiplier

        policy_id = f"POL{secrets.token_hex(4).upper()}"
        policy = {
            'policy_id': policy_id,
            'employee_id': employee_id,
            'coverage_id': coverage_id,
            'coverage_details': coverage,
            'effective_date': effective_date,
            'monthly_premium': round(final_premium, 2),
            'annual_premium': round(final_premium * 12, 2),
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        insurance_policies[policy_id] = policy

        return jsonify({
            'status': 'success',
            'policy': policy,
            'message': 'Insurance policy created successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@insurance_bp.route('/policies/<policy_id>', methods=['GET'])
@insurance_token_required
def get_policy(policy_id):
    """Get policy details"""
    try:
        policy = insurance_policies.get(policy_id)
        if not policy:
            return jsonify({'error': 'Policy not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'policy': policy,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@insurance_bp.route('/policies/<policy_id>/cancel', methods=['POST'])
@insurance_token_required
def cancel_policy(policy_id):
    """Cancel insurance policy"""
    try:
        data = request.get_json(force=True)
        cancellation_reason = data.get('reason', 'Not specified')

        policy = insurance_policies.get(policy_id)
        if not policy:
            return jsonify({'error': 'Policy not found', 'status': 'error'}), 404

        if policy['status'] != 'active':
            return jsonify({'error': 'Policy is not active', 'status': 'error'}), 400

        policy['status'] = 'cancelled'
        policy['cancellation_date'] = datetime.now(timezone.utc).date().isoformat()
        policy['cancellation_reason'] = cancellation_reason
        policy['updated_at'] = datetime.now(timezone.utc).isoformat()

        return jsonify({
            'status': 'success',
            'policy': policy,
            'message': 'Policy cancelled successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Claims Management Endpoints
@insurance_bp.route('/claims', methods=['GET'])
@insurance_token_required
def get_claims():
    """Get all insurance claims"""
    try:
        claims = list(insurance_claims.values())
        return jsonify({
            'status': 'success',
            'claims': claims,
            'count': len(claims),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@insurance_bp.route('/claims', methods=['POST'])
@insurance_token_required
def submit_claim():
    """Submit new insurance claim"""
    try:
        data = request.get_json(force=True)
        policy_id = data.get('policy_id')
        claim_type = data.get('claim_type')
        incident_date = data.get('incident_date')
        amount = data.get('amount')
        description = data.get('description', '')
        service_provider = data.get('service_provider', '')

        if not all([policy_id, claim_type, incident_date, amount]):
            return jsonify({'error': 'Policy ID, claim type, incident date, and amount are required', 'status': 'error'}), 400

        if policy_id not in insurance_policies:
            return jsonify({'error': 'Policy not found', 'status': 'error'}), 404

        policy = insurance_policies[policy_id]
        if policy['status'] != 'active':
            return jsonify({'error': 'Policy is not active', 'status': 'error'}), 400

        # Validate claim amount against coverage
        coverage = policy['coverage_details']
        if amount > coverage.get('annual_limit', float('inf')):
            return jsonify({'error': 'Claim amount exceeds coverage limit', 'status': 'error'}), 400

        claim_id = f"CLM{secrets.token_hex(4).upper()}"
        claim = {
            'claim_id': claim_id,
            'policy_id': policy_id,
            'employee_id': policy['employee_id'],
            'claim_type': claim_type,
            'incident_date': incident_date,
            'amount_requested': float(amount),
            'description': description,
            'service_provider': service_provider,
            'status': 'submitted',
            'submitted_date': datetime.now(timezone.utc).date().isoformat(),
            'created_at': datetime.now(timezone.utc).isoformat()
        }

        insurance_claims[claim_id] = claim

        return jsonify({
            'status': 'success',
            'claim': claim,
            'message': 'Insurance claim submitted successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 201
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@insurance_bp.route('/claims/<claim_id>', methods=['GET'])
@insurance_token_required
def get_claim(claim_id):
    """Get claim details"""
    try:
        claim = insurance_claims.get(claim_id)
        if not claim:
            return jsonify({'error': 'Claim not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'claim': claim,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@insurance_bp.route('/claims/<claim_id>/review', methods=['POST'])
@insurance_token_required
def review_claim(claim_id):
    """Review and process insurance claim"""
    try:
        data = request.get_json(force=True)
        decision = data.get('decision')  # 'approve' or 'deny'
        approved_amount = data.get('approved_amount')
        denial_reason = data.get('denial_reason', '')
        reviewer_notes = data.get('reviewer_notes', '')

        if decision not in ['approve', 'deny']:
            return jsonify({'error': 'Decision must be approve or deny', 'status': 'error'}), 400

        claim = insurance_claims.get(claim_id)
        if not claim:
            return jsonify({'error': 'Claim not found', 'status': 'error'}), 404

        if claim['status'] != 'submitted':
            return jsonify({'error': 'Claim is not in submitted status', 'status': 'error'}), 400

        claim['status'] = 'approved' if decision == 'approve' else 'denied'
        claim['review_date'] = datetime.now(timezone.utc).date().isoformat()
        claim['reviewer_notes'] = reviewer_notes

        if decision == 'approve':
            if not approved_amount:
                return jsonify({'error': 'Approved amount is required for approval', 'status': 'error'}), 400
            claim['amount_approved'] = float(approved_amount)
            claim['amount_paid'] = float(approved_amount)
            claim['payment_date'] = datetime.now(timezone.utc).date().isoformat()
        else:
            claim['denial_reason'] = denial_reason

        claim['updated_at'] = datetime.now(timezone.utc).isoformat()

        return jsonify({
            'status': 'success',
            'claim': claim,
            'message': f'Claim {decision}d successfully',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Coverage Types Endpoints
@insurance_bp.route('/coverage-types', methods=['GET'])
@insurance_token_required
def get_coverage_types():
    """Get all coverage types"""
    try:
        types = list(coverage_types.values())
        return jsonify({
            'status': 'success',
            'coverage_types': types,
            'count': len(types),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@insurance_bp.route('/coverage-types/<coverage_id>', methods=['GET'])
@insurance_token_required
def get_coverage_type(coverage_id):
    """Get specific coverage type"""
    try:
        coverage = coverage_types.get(coverage_id)
        if not coverage:
            return jsonify({'error': 'Coverage type not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'coverage_type': coverage,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Underwriting Endpoints
@insurance_bp.route('/underwriting/quote', methods=['POST'])
@insurance_token_required
def get_underwriting_quote():
    """Get underwriting quote for coverage"""
    try:
        data = request.get_json(force=True)
        coverage_id = data.get('coverage_id')
        age = data.get('age', 35)
        health_status = data.get('health_status', 'good')  # excellent, good, fair, poor
        smoker = data.get('smoker', False)
        occupation_risk = data.get('occupation_risk', 'low')  # low, medium, high

        if not coverage_id:
            return jsonify({'error': 'Coverage ID is required', 'status': 'error'}), 400

        if coverage_id not in coverage_types:
            return jsonify({'error': 'Invalid coverage type', 'status': 'error'}), 404

        coverage = coverage_types[coverage_id]

        # Calculate premium based on underwriting factors
        base_premium = coverage['monthly_premium']

        # Age factor
        age_factor = 1.0
        if age < 30:
            age_factor = 0.8
        elif age > 50:
            age_factor = 1.8
        elif age > 40:
            age_factor = 1.3

        # Health factor
        health_factors = {
            'excellent': 0.7,
            'good': 1.0,
            'fair': 1.4,
            'poor': 2.0
        }
        health_factor = health_factors.get(health_status, 1.0)

        # Smoking factor
        smoking_factor = 1.5 if smoker else 1.0

        # Occupation risk factor
        occupation_factors = {
            'low': 1.0,
            'medium': 1.2,
            'high': 1.5
        }
        occupation_factor = occupation_factors.get(occupation_risk, 1.0)

        # Calculate final premium
        final_premium = base_premium * age_factor * health_factor * smoking_factor * occupation_factor

        quote = {
            'coverage_id': coverage_id,
            'coverage_name': coverage['name'],
            'base_premium': base_premium,
            'final_monthly_premium': round(final_premium, 2),
            'final_annual_premium': round(final_premium * 12, 2),
            'factors_applied': {
                'age_factor': round(age_factor, 2),
                'health_factor': round(health_factor, 2),
                'smoking_factor': round(smoking_factor, 2),
                'occupation_factor': round(occupation_factor, 2)
            },
            'underwriting_details': {
                'age': age,
                'health_status': health_status,
                'smoker': smoker,
                'occupation_risk': occupation_risk
            }
        }

        return jsonify({
            'status': 'success',
            'quote': quote,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Analytics Endpoints
@insurance_bp.route('/analytics/claims', methods=['GET'])
@insurance_token_required
def get_claims_analytics():
    """Get claims analytics"""
    try:
        claims = list(insurance_claims.values())

        if not claims:
            return jsonify({
                'status': 'success',
                'analytics': {
                    'total_claims': 0,
                    'approved_claims': 0,
                    'denied_claims': 0,
                    'total_amount_requested': 0,
                    'total_amount_paid': 0,
                    'average_processing_time': 0
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200

        total_claims = len(claims)
        approved_claims = len([c for c in claims if c.get('status') == 'approved'])
        denied_claims = len([c for c in claims if c.get('status') == 'denied'])

        total_requested = sum(c.get('amount_requested', 0) for c in claims)
        total_paid = sum(c.get('amount_paid', 0) for c in claims)

        # Calculate average processing time (simplified)
        avg_processing_time = 3.5  # days

        analytics = {
            'total_claims': total_claims,
            'approved_claims': approved_claims,
            'denied_claims': denied_claims,
            'approval_rate': round((approved_claims / total_claims) * 100, 2) if total_claims > 0 else 0,
            'total_amount_requested': round(total_requested, 2),
            'total_amount_paid': round(total_paid, 2),
            'average_claim_amount': round(total_requested / total_claims, 2) if total_claims > 0 else 0,
            'average_processing_time_days': avg_processing_time
        }

        return jsonify({
            'status': 'success',
            'analytics': analytics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@insurance_bp.route('/analytics/policies', methods=['GET'])
@insurance_token_required
def get_policies_analytics():
    """Get policies analytics"""
    try:
        policies = list(insurance_policies.values())

        if not policies:
            return jsonify({
                'status': 'success',
                'analytics': {
                    'total_policies': 0,
                    'active_policies': 0,
                    'total_monthly_premium': 0,
                    'total_annual_premium': 0
                },
                'timestamp': datetime.now(timezone.utc).isoformat()
            }), 200

        total_policies = len(policies)
        active_policies = len([p for p in policies if p.get('status') == 'active'])

        total_monthly = sum(p.get('monthly_premium', 0) for p in policies if p.get('status') == 'active')
        total_annual = sum(p.get('annual_premium', 0) for p in policies if p.get('status') == 'active')

        analytics = {
            'total_policies': total_policies,
            'active_policies': active_policies,
            'cancelled_policies': total_policies - active_policies,
            'total_monthly_premium': round(total_monthly, 2),
            'total_annual_premium': round(total_annual, 2),
            'average_monthly_premium': round(total_monthly / active_policies, 2) if active_policies > 0 else 0
        }

        return jsonify({
            'status': 'success',
            'analytics': analytics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Export functions for integration
def get_insurance_blueprint():
    """Get the insurance blueprint for integration"""
    return insurance_bp

def get_insurance_endpoints():
    """Get list of insurance endpoints for documentation"""
    return [
        '/api/insurance/policies - Policy management',
        '/api/insurance/claims - Claims management',
        '/api/insurance/coverage-types - Coverage type management',
        '/api/insurance/underwriting/quote - Underwriting quotes',
        '/api/insurance/analytics/claims - Claims analytics',
        '/api/insurance/analytics/policies - Policy analytics'
    ]
