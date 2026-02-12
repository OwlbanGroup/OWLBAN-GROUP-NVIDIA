"""
Business Blueprint for JPMorgan Financial APIs
Provides business management functionality including CRUD operations for business entities.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional, List

# Import services and utilities
from src.logger import telemetry_logger

# Import authentication and rate limiting decorators
try:
    from src.auth import token_auth_required
    from src.rate_limiting import conditional_limit
except ImportError:
    # Fallback if not found - these would need to be implemented
    def token_auth_required(f):
        return f
    def conditional_limit(rate):
        def decorator(f):
            return f
        return decorator

business_bp = Blueprint('business', __name__)

# Mock business storage (in real implementation, this would be a database)
_mock_businesses = {}


# =============================================================================
# BUSINESS CRUD ENDPOINTS
# =============================================================================

@business_bp.route('/businesses', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def create_business():
    """
    Create a new business
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token
        name = data.get('name')
        business_type = data.get('type')
        description = data.get('description', '')
        address = data.get('address', {})
        contact_info = data.get('contact_info', {})

        if not name or not business_type:
            return jsonify({'error': 'Business name and type are required', 'status': 'error'}), 400

        # Validate business type
        valid_types = ['corporation', 'llc', 'partnership', 'sole_proprietorship', 'nonprofit']
        if business_type not in valid_types:
            return jsonify({'error': f'Invalid business type. Must be one of: {valid_types}', 'status': 'error'}), 400

        # Create business
        business_id = str(uuid.uuid4())
        business = {
            'id': business_id,
            'user_id': user_id,
            'name': name,
            'type': business_type,
            'description': description,
            'address': address,
            'contact_info': contact_info,
            'is_active': True,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

        _mock_businesses[business_id] = business

        telemetry_logger.log_info(f"Business created: {business_id} for user {user_id}")

        return jsonify({
            'status': 'success',
            'message': 'Business created successfully',
            'business': business
        }), 201

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_business'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@business_bp.route('/businesses', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def list_businesses():
    """
    List all businesses for the authenticated user
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Parse query parameters
        limit = min(int(request.args.get('limit', 50)), 100)
        offset = int(request.args.get('offset', 0))
        business_type = request.args.get('type')
        active_only = request.args.get('active_only', 'true').lower() == 'true'

        # Get user businesses
        user_businesses = [
            b for b in _mock_businesses.values()
            if b['user_id'] == user_id
        ]

        # Filter by type if provided
        if business_type:
            user_businesses = [
                b for b in user_businesses
                if b.get('type') == business_type
            ]

        # Filter by active status if requested
        if active_only:
            user_businesses = [
                b for b in user_businesses
                if b.get('is_active', True)
            ]

        # Apply pagination
        paginated_businesses = user_businesses[offset:offset + limit]

        return jsonify({
            'status': 'success',
            'businesses': paginated_businesses,
            'count': len(paginated_businesses),
            'total_count': len(user_businesses)
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_businesses'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@business_bp.route('/businesses/<business_id>', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_business(business_id):
    """
    Get specific business details
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        business = _mock_businesses.get(business_id)

        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404

        # Check ownership
        if business['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        return jsonify({
            'status': 'success',
            'business': business
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_business'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@business_bp.route('/businesses/<business_id>', methods=['PUT'])
@token_auth_required
@conditional_limit("10 per minute")
def update_business(business_id):
    """
    Update business information
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        business = _mock_businesses.get(business_id)

        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404

        # Check ownership
        if business['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        # Validate business type if provided
        if 'type' in data:
            valid_types = ['corporation', 'llc', 'partnership', 'sole_proprietorship', 'nonprofit']
            if data['type'] not in valid_types:
                return jsonify({'error': f'Invalid business type. Must be one of: {valid_types}', 'status': 'error'}), 400

        # Update business
        business.update({
            'name': data.get('name', business['name']),
            'type': data.get('type', business['type']),
            'description': data.get('description', business['description']),
            'address': data.get('address', business['address']),
            'contact_info': data.get('contact_info', business['contact_info']),
            'is_active': data.get('is_active', business['is_active']),
            'updated_at': datetime.now(timezone.utc).isoformat()
        })

        telemetry_logger.log_info(f"Business updated: {business_id}")

        return jsonify({
            'status': 'success',
            'message': 'Business updated successfully',
            'business': business
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_business'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@business_bp.route('/businesses/<business_id>', methods=['DELETE'])
@token_auth_required
@conditional_limit("5 per minute")
def delete_business(business_id):
    """
    Delete a business
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        business = _mock_businesses.get(business_id)

        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404

        # Check ownership
        if business['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        # Delete business
        del _mock_businesses[business_id]

        telemetry_logger.log_info(f"Business deleted: {business_id}")

        return jsonify({
            'status': 'success',
            'message': 'Business deleted successfully'
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'delete_business'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# BUSINESS ANALYTICS ENDPOINTS
# =============================================================================

@business_bp.route('/businesses/<business_id>/analytics', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_business_analytics(business_id):
    """
    Get business analytics and insights
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        business = _mock_businesses.get(business_id)

        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404

        # Check ownership
        if business['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        # Mock analytics data
        analytics = {
            'revenue_trend': [12000, 15000, 18000, 22000, 25000, 28000],
            'expense_trend': [8000, 9500, 11000, 13000, 14000, 16000],
            'profit_margin': 0.35,
            'growth_rate': 0.15,
            'key_metrics': {
                'total_revenue': 120000,
                'total_expenses': 83000,
                'net_profit': 37000,
                'employee_count': 25,
                'customer_count': 150
            },
            'period': 'last_6_months'
        }

        return jsonify({
            'status': 'success',
            'business_id': business_id,
            'analytics': analytics
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_business_analytics'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@business_bp.route('/businesses/<business_id>/financials', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_business_financials(business_id):
    """
    Get business financial information
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        business = _mock_businesses.get(business_id)

        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404

        # Check ownership
        if business['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        # Mock financial data
        financials = {
            'balance_sheet': {
                'assets': {
                    'current_assets': 45000,
                    'fixed_assets': 120000,
                    'total_assets': 165000
                },
                'liabilities': {
                    'current_liabilities': 25000,
                    'long_term_debt': 75000,
                    'total_liabilities': 100000
                },
                'equity': {
                    'owner_equity': 65000
                }
            },
            'income_statement': {
                'revenue': 120000,
                'cost_of_goods_sold': 48000,
                'gross_profit': 72000,
                'operating_expenses': 35000,
                'net_income': 37000
            },
            'cash_flow': {
                'operating_cash_flow': 42000,
                'investing_cash_flow': -15000,
                'financing_cash_flow': -8000,
                'net_cash_flow': 19000
            },
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

        return jsonify({
            'status': 'success',
            'business_id': business_id,
            'financials': financials
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_business_financials'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# BUSINESS COMPLIANCE ENDPOINTS
# =============================================================================

@business_bp.route('/businesses/<business_id>/compliance', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_business_compliance(business_id):
    """
    Get business compliance status
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        business = _mock_businesses.get(business_id)

        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404

        # Check ownership
        if business['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        # Mock compliance data
        compliance = {
            'tax_compliance': {
                'status': 'compliant',
                'last_filing': '2024-01-15',
                'next_due': '2024-04-15',
                'issues': []
            },
            'regulatory_compliance': {
                'status': 'compliant',
                'licenses': [
                    {'type': 'business_license', 'status': 'active', 'expires': '2025-01-01'},
                    {'type': 'industry_certification', 'status': 'active', 'expires': '2024-08-15'}
                ]
            },
            'insurance_compliance': {
                'status': 'compliant',
                'policies': [
                    {'type': 'liability', 'provider': 'ABC Insurance', 'expires': '2024-06-01'},
                    {'type': 'property', 'provider': 'XYZ Insurance', 'expires': '2024-09-01'}
                ]
            },
            'overall_status': 'compliant',
            'last_review': datetime.now(timezone.utc).isoformat()
        }

        return jsonify({
            'status': 'success',
            'business_id': business_id,
            'compliance': compliance
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_business_compliance'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# BUSINESS REPORTING ENDPOINTS
# =============================================================================

@business_bp.route('/businesses/<business_id>/reports', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_business_reports(business_id):
    """
    Get business reports
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        business = _mock_businesses.get(business_id)

        if not business:
            return jsonify({'error': 'Business not found', 'status': 'error'}), 404

        # Check ownership
        if business['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        # Parse query parameters
        report_type = request.args.get('type', 'financial')
        period = request.args.get('period', 'monthly')

        # Mock reports data
        reports = {
            'available_reports': [
                {'type': 'financial', 'period': 'monthly', 'generated_at': '2024-01-31'},
                {'type': 'financial', 'period': 'quarterly', 'generated_at': '2024-01-31'},
                {'type': 'tax', 'period': 'annual', 'generated_at': '2024-01-15'},
                {'type': 'compliance', 'period': 'quarterly', 'generated_at': '2024-01-31'}
            ],
            'requested_report': {
                'type': report_type,
                'period': period,
                'data': {
                    'summary': f'{report_type.title()} report for {business["name"]} - {period}',
                    'generated_at': datetime.now(timezone.utc).isoformat(),
                    'key_findings': ['Revenue increased by 15%', 'Expenses within budget', 'Profit margin improved']
                }
            }
        }

        return jsonify({
            'status': 'success',
            'business_id': business_id,
            'reports': reports
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_business_reports'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# BUSINESS DASHBOARD ENDPOINTS
# =============================================================================

@business_bp.route('/dashboard', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_businesses_dashboard():
    """
    Get businesses dashboard overview
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Get user businesses
        user_businesses = [
            b for b in _mock_businesses.values()
            if b['user_id'] == user_id
        ]

        # Calculate dashboard stats
        total_businesses = len(user_businesses)
        active_businesses = len([b for b in user_businesses if b.get('is_active', True)])
        total_revenue = sum(100000 for _ in user_businesses)  # Mock revenue per business
        avg_growth = 0.12  # Mock growth rate

        # Group by type
        type_distribution = {}
        for business in user_businesses:
            business_type = business.get('type', 'other')
            type_distribution[business_type] = type_distribution.get(business_type, 0) + 1

        # Recent businesses
        recent_businesses = sorted(
            user_businesses,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )[:5]

        # Performance metrics
        performance = {
            'revenue_trend': [85000, 92000, 101000, 118000, 125000],
            'profit_trend': [15000, 18000, 22000, 28000, 32000],
            'growth_rate': avg_growth
        }

        dashboard = {
            'stats': {
                'total_businesses': total_businesses,
                'active_businesses': active_businesses,
                'total_revenue': total_revenue,
                'avg_growth_rate': avg_growth
            },
            'type_distribution': type_distribution,
            'recent_businesses': [
                {
                    'id': b['id'],
                    'name': b['name'],
                    'type': b['type'],
                    'created_at': b['created_at']
                }
                for b in recent_businesses
            ],
            'performance': performance,
            'alerts': [
                {'type': 'tax_filing', 'message': 'Q4 tax filing due in 15 days', 'priority': 'high'},
                {'type': 'compliance', 'message': 'License renewal due next month', 'priority': 'medium'}
            ]
        }

        return jsonify({
            'status': 'success',
            'dashboard': dashboard
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_businesses_dashboard'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500
