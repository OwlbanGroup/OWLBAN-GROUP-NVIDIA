"""
Asset Blueprint for JPMorgan Financial APIs
Provides asset management functionality including CRUD operations for financial assets.
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

asset_bp = Blueprint('asset', __name__)

# Mock asset storage (in real implementation, this would be a database)
_mock_assets = {}


# =============================================================================
# ASSET CRUD ENDPOINTS
# =============================================================================

@asset_bp.route('/assets', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def create_asset():
    """
    Create a new asset
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token
        business_id = data.get('business_id')
        name = data.get('name')
        asset_type = data.get('type')
        value = data.get('value', 0.0)
        description = data.get('description', '')

        if not name or not asset_type:
            return jsonify({'error': 'Asset name and type are required', 'status': 'error'}), 400

        # Validate asset type
        valid_types = ['real_estate', 'equipment', 'inventory', 'securities', 'cash', 'other']
        if asset_type not in valid_types:
            return jsonify({'error': f'Invalid asset type. Must be one of: {valid_types}', 'status': 'error'}), 400

        # Create asset
        asset_id = str(uuid.uuid4())
        asset = {
            'id': asset_id,
            'user_id': user_id,
            'business_id': business_id,
            'name': name,
            'type': asset_type,
            'value': value,
            'description': description,
            'is_active': True,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

        _mock_assets[asset_id] = asset

        telemetry_logger.log_info(f"Asset created: {asset_id} for user {user_id}")

        return jsonify({
            'status': 'success',
            'message': 'Asset created successfully',
            'asset': asset
        }), 201

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_asset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@asset_bp.route('/assets', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def list_assets():
    """
    List all assets for the authenticated user
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Parse query parameters
        limit = min(int(request.args.get('limit', 50)), 100)
        offset = int(request.args.get('offset', 0))
        asset_type = request.args.get('type')
        business_id = request.args.get('business_id')

        # Get user assets
        user_assets = [
            a for a in _mock_assets.values()
            if a['user_id'] == user_id
        ]

        # Filter by type if provided
        if asset_type:
            user_assets = [
                a for a in user_assets
                if a.get('type') == asset_type
            ]

        # Filter by business if provided
        if business_id:
            user_assets = [
                a for a in user_assets
                if a.get('business_id') == business_id
            ]

        # Apply pagination
        paginated_assets = user_assets[offset:offset + limit]

        return jsonify({
            'status': 'success',
            'assets': paginated_assets,
            'count': len(paginated_assets),
            'total_count': len(user_assets)
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_assets'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@asset_bp.route('/assets/<asset_id>', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_asset(asset_id):
    """
    Get specific asset details
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        asset = _mock_assets.get(asset_id)

        if not asset:
            return jsonify({'error': 'Asset not found', 'status': 'error'}), 404

        # Check ownership
        if asset['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        return jsonify({
            'status': 'success',
            'asset': asset
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_asset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@asset_bp.route('/assets/<asset_id>', methods=['PUT'])
@token_auth_required
@conditional_limit("10 per minute")
def update_asset(asset_id):
    """
    Update asset information
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        asset = _mock_assets.get(asset_id)

        if not asset:
            return jsonify({'error': 'Asset not found', 'status': 'error'}), 404

        # Check ownership
        if asset['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        # Validate asset type if provided
        if 'type' in data:
            valid_types = ['real_estate', 'equipment', 'inventory', 'securities', 'cash', 'other']
            if data['type'] not in valid_types:
                return jsonify({'error': f'Invalid asset type. Must be one of: {valid_types}', 'status': 'error'}), 400

        # Update asset
        asset.update({
            'name': data.get('name', asset['name']),
            'type': data.get('type', asset['type']),
            'value': data.get('value', asset['value']),
            'description': data.get('description', asset['description']),
            'business_id': data.get('business_id', asset['business_id']),
            'updated_at': datetime.now(timezone.utc).isoformat()
        })

        telemetry_logger.log_info(f"Asset updated: {asset_id}")

        return jsonify({
            'status': 'success',
            'message': 'Asset updated successfully',
            'asset': asset
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_asset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@asset_bp.route('/assets/<asset_id>', methods=['DELETE'])
@token_auth_required
@conditional_limit("5 per minute")
def delete_asset(asset_id):
    """
    Delete an asset
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        asset = _mock_assets.get(asset_id)

        if not asset:
            return jsonify({'error': 'Asset not found', 'status': 'error'}), 404

        # Check ownership
        if asset['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        # Delete asset
        del _mock_assets[asset_id]

        telemetry_logger.log_info(f"Asset deleted: {asset_id}")

        return jsonify({
            'status': 'success',
            'message': 'Asset deleted successfully'
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'delete_asset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# ASSET VALUATION ENDPOINTS
# =============================================================================

@asset_bp.route('/assets/<asset_id>/valuation', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_asset_valuation(asset_id):
    """
    Get asset valuation data
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        asset = _mock_assets.get(asset_id)

        if not asset:
            return jsonify({'error': 'Asset not found', 'status': 'error'}), 404

        # Check ownership
        if asset['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        # Mock valuation data
        valuation = {
            'current_value': asset['value'],
            'appraised_value': asset['value'] * 1.05,
            'market_value': asset['value'] * 0.98,
            'depreciation_rate': 0.05,
            'last_appraisal': datetime.now(timezone.utc).isoformat(),
            'next_appraisal_due': (datetime.now(timezone.utc).replace(year=datetime.now().year + 1)).isoformat()
        }

        return jsonify({
            'status': 'success',
            'asset_id': asset_id,
            'valuation': valuation
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_asset_valuation'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@asset_bp.route('/assets/<asset_id>/depreciation', methods=['GET'])
@token_auth_required
@conditional_limit("15 per minute")
def get_asset_depreciation(asset_id):
    """
    Get asset depreciation information
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        asset = _mock_assets.get(asset_id)

        if not asset:
            return jsonify({'error': 'Asset not found', 'status': 'error'}), 404

        # Check ownership
        if asset['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        # Mock depreciation data
        depreciation = {
            'original_value': asset['value'],
            'current_value': asset['value'] * 0.85,
            'accumulated_depreciation': asset['value'] * 0.15,
            'depreciation_method': 'straight_line',
            'useful_life_years': 10,
            'years_used': 1.5,
            'remaining_life_years': 8.5
        }

        return jsonify({
            'status': 'success',
            'asset_id': asset_id,
            'depreciation': depreciation
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_asset_depreciation'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# ASSET PORTFOLIO ENDPOINTS
# =============================================================================

@asset_bp.route('/portfolio', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_asset_portfolio():
    """
    Get user's asset portfolio overview
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Get user assets
        user_assets = [
            a for a in _mock_assets.values()
            if a['user_id'] == user_id
        ]

        # Calculate portfolio stats
        total_value = sum(a.get('value', 0) for a in user_assets)
        asset_types = {}
        for asset in user_assets:
            asset_type = asset.get('type', 'other')
            if asset_type not in asset_types:
                asset_types[asset_type] = {'count': 0, 'value': 0}
            asset_types[asset_type]['count'] += 1
            asset_types[asset_type]['value'] += asset.get('value', 0)

        portfolio = {
            'total_assets': len(user_assets),
            'total_value': total_value,
            'asset_types': asset_types,
            'top_assets': sorted(user_assets, key=lambda x: x.get('value', 0), reverse=True)[:5]
        }

        return jsonify({
            'status': 'success',
            'portfolio': portfolio
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_asset_portfolio'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@asset_bp.route('/portfolio/performance', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_portfolio_performance():
    """
    Get portfolio performance metrics
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Get user assets
        user_assets = [
            a for a in _mock_assets.values()
            if a['user_id'] == user_id
        ]

        # Mock performance data
        performance = {
            'total_return': 12.5,
            'annual_return': 8.3,
            'volatility': 15.2,
            'sharpe_ratio': 0.55,
            'max_drawdown': -8.7,
            'period': '1_year',
            'benchmark_comparison': 2.1  # Percentage above benchmark
        }

        return jsonify({
            'status': 'success',
            'performance': performance
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_portfolio_performance'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# ASSET TRANSFER ENDPOINTS
# =============================================================================

@asset_bp.route('/assets/<asset_id>/transfer', methods=['POST'])
@token_auth_required
@conditional_limit("5 per minute")
def transfer_asset(asset_id):
    """
    Transfer asset ownership
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        asset = _mock_assets.get(asset_id)

        if not asset:
            return jsonify({'error': 'Asset not found', 'status': 'error'}), 404

        # Check ownership
        if asset['user_id'] != user_id:
            return jsonify({'error': 'Access denied', 'status': 'error'}), 403

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        new_owner_id = data.get('new_owner_id')
        transfer_reason = data.get('reason', '')

        if not new_owner_id:
            return jsonify({'error': 'New owner ID is required', 'status': 'error'}), 400

        # Transfer asset
        asset['user_id'] = new_owner_id
        asset['transfer_history'] = asset.get('transfer_history', [])
        asset['transfer_history'].append({
            'from_user': user_id,
            'to_user': new_owner_id,
            'reason': transfer_reason,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        asset['updated_at'] = datetime.now(timezone.utc).isoformat()

        telemetry_logger.log_info(f"Asset transferred: {asset_id} from {user_id} to {new_owner_id}")

        return jsonify({
            'status': 'success',
            'message': 'Asset transferred successfully',
            'asset': asset
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'transfer_asset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# ASSET DASHBOARD ENDPOINTS
# =============================================================================

@asset_bp.route('/dashboard', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_assets_dashboard():
    """
    Get assets dashboard overview
    """
@asset_bp.route('/dashboard', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_assets_dashboard():
    """
    Get assets dashboard overview
    """
    try:
        user_id = getattr(request, 'user_id', 'demo_user')  # Would come from JWT token

        # Get user assets
        user_assets = [
            a for a in _mock_assets.values()
            if a['user_id'] == user_id
        ]

        # Calculate dashboard stats
        total_assets = len(user_assets)
        total_value = sum(a.get('value', 0) for a in user_assets)
        active_assets = len([a for a in user_assets if a.get('is_active', True)])

        # Group by type
        type_distribution = {}
        for asset in user_assets:
            asset_type = asset.get('type', 'other')
            type_distribution[asset_type] = type_distribution.get(asset_type, 0) + 1

        # Recent assets
        recent_assets = sorted(
            user_assets,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )[:5]

        dashboard_data = {
            'stats': {
                'total_assets': total_assets,
                'active_assets': active_assets,
                'total_value': total_value,
                'average_value': total_value / max(total_assets, 1)
            },
            'type_distribution': type_distribution,
            'recent_assets': recent_assets,
            'high_value_assets': sorted(user_assets, key=lambda x: x.get('value', 0), reverse=True)[:3]
        }

        return jsonify({
            'status': 'success',
            'dashboard': dashboard_data
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_assets_dashboard'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


        # Calculate dashboard stats
        total_assets = len(user_assets)
        total_value = sum(a.get('value', 0) for a in user_assets)
        active_assets = len([a for a in user_assets if a.get('is_active', True)])

        # Group by type
        type_distribution = {}
        for asset in user_assets:
            asset_type = asset.get('type', 'other')
            type_distribution[asset_type] = type_distribution.get(asset_type, 0) + 1

        # Recent assets
        recent_assets = sorted(
            user_assets,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )[:5]

        dashboard_data = {
            'stats': {
                'total_assets': total_assets,
                'active_assets': active_assets,
                'total_value': total_value,
                'average_value': total_value / max(total_assets, 1)
            },
            'type_distribution': type_distribution,
            'recent_assets': recent_assets,
            'high_value_assets': sorted(user_assets, key=lambda x: x.get('value', 0), reverse=True)[:3]
        }

        return jsonify({
            'status': 'success',
            'dashboard': dashboard_data
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_assets_dashboard'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500
