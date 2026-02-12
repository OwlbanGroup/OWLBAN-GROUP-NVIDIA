"""
Data Blueprint for JPMorgan Financial APIs
Provides data processing, conversion, and management functionality.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, Optional, List

# Import services and utilities
from src.data_format_converter import DataFormatConverter
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

data_bp = Blueprint('data', __name__)

# Mock data storage (in real implementation, this would be a database)
_mock_data_sets = {}


# =============================================================================
# DATA CONVERSION ENDPOINTS
# =============================================================================

@data_bp.route('/data/convert', methods=['POST'])
@token_auth_required
@conditional_limit("15 per minute")
def convert_data():
    """
    Convert data between different formats
    """
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided for conversion', 'status': 'error'}), 400

        input_data = data['data']
        from_format = data.get('from_format', 'json').lower()
        to_format = data.get('to_format', 'json').lower()
        options = data.get('options', {})

        if not isinstance(input_data, list):
            return jsonify({'error': 'Data must be a list of records', 'status': 'error'}), 400

        # Validate formats
        if from_format not in DataFormatConverter.get_supported_import_formats():
            return jsonify({
                'error': f'Unsupported input format. Supported: {DataFormatConverter.get_supported_import_formats()}',
                'status': 'error'
            }), 400

        if to_format not in DataFormatConverter.get_supported_formats():
            return jsonify({
                'error': f'Unsupported output format. Supported: {DataFormatConverter.get_supported_formats()}',
                'status': 'error'
            }), 400

        # Convert data
        if to_format == 'json':
            result = DataFormatConverter.convert_to_json(input_data, pretty=options.get('pretty', True))
            content_type = 'application/json'
        elif to_format == 'csv':
            result = DataFormatConverter.convert_to_csv(input_data)
            content_type = 'text/csv'
        elif to_format == 'xml':
            result = DataFormatConverter.convert_to_xml(input_data)
            content_type = 'application/xml'
        elif to_format == 'yaml':
            result = DataFormatConverter.convert_to_yaml(input_data)
            content_type = 'application/x-yaml'
        elif to_format == 'excel':
            result_bytes = DataFormatConverter.convert_to_excel(input_data)
            return result_bytes, 200, {'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
        elif to_format == 'parquet':
            result_bytes = DataFormatConverter.convert_to_parquet(input_data)
            return result_bytes, 200, {'Content-Type': 'application/octet-stream'}
        else:
            return jsonify({
                'error': f'Unsupported conversion to {to_format}',
                'status': 'error'
            }), 400

        return result, 200, {'Content-Type': content_type}

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'convert_data'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@data_bp.route('/data/formats', methods=['GET'])
@token_auth_required
@conditional_limit("30 per minute")
def get_supported_formats():
    """
    Get list of supported data formats
    """
    try:
        return jsonify({
            'status': 'success',
            'import_formats': DataFormatConverter.get_supported_import_formats(),
            'export_formats': DataFormatConverter.get_supported_formats(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_supported_formats'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# DATA VALIDATION ENDPOINTS
# =============================================================================

@data_bp.route('/data/validate', methods=['POST'])
@token_auth_required
@conditional_limit("20 per minute")
def validate_data():
    """
    Validate data against a schema
    """
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided for validation', 'status': 'error'}), 400

        input_data = data['data']
        schema_type = data.get('schema_type', 'generic')
        strict_mode = data.get('strict_mode', False)

        if not isinstance(input_data, list):
            return jsonify({'error': 'Data must be a list of records', 'status': 'error'}), 400

        # Validate data (mock validation for demo)
        validation_results = {
            'is_valid': True,
            'total_records': len(input_data),
            'valid_records': len(input_data),
            'invalid_records': 0,
            'errors': [],
            'warnings': []
        }

        # Basic validation checks
        for i, record in enumerate(input_data):
            if not isinstance(record, dict):
                validation_results['is_valid'] = False
                validation_results['invalid_records'] += 1
                validation_results['valid_records'] -= 1
                validation_results['errors'].append({
                    'record_index': i,
                    'error': 'Record must be a dictionary'
                })

        return jsonify({
            'status': 'success',
            'validation': validation_results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'validate_data'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# DATA TRANSFORMATION ENDPOINTS
# =============================================================================

@data_bp.route('/data/transform', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def transform_data():
    """
    Transform data using various operations
    """
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided for transformation', 'status': 'error'}), 400

        input_data = data['data']
        transformations = data.get('transformations', [])

        if not isinstance(input_data, list):
            return jsonify({'error': 'Data must be a list of records', 'status': 'error'}), 400

        if not isinstance(transformations, list):
            return jsonify({'error': 'Transformations must be a list', 'status': 'error'}), 400

        # Apply transformations (mock for demo)
        transformed_data = input_data.copy()
        applied_transformations = []

        for transformation in transformations:
            transform_type = transformation.get('type')
            if transform_type == 'filter':
                # Mock filtering
                condition = transformation.get('condition', {})
                transformed_data = [r for r in transformed_data if True]  # Mock filter
                applied_transformations.append(f"Applied filter: {condition}")
            elif transform_type == 'sort':
                # Mock sorting
                field = transformation.get('field', 'id')
                order = transformation.get('order', 'asc')
                transformed_data.sort(key=lambda x: x.get(field, ''), reverse=(order == 'desc'))
                applied_transformations.append(f"Sorted by {field} ({order})")
            elif transform_type == 'aggregate':
                # Mock aggregation
                group_by = transformation.get('group_by', [])
                operations = transformation.get('operations', [])
                # Mock aggregation result
                transformed_data = [{'group': 'mock_group', 'count': len(transformed_data)}]
                applied_transformations.append(f"Aggregated by {group_by}")

        return jsonify({
            'status': 'success',
            'original_count': len(input_data),
            'transformed_count': len(transformed_data),
            'transformations_applied': applied_transformations,
            'data': transformed_data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'transform_data'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# DATASET MANAGEMENT ENDPOINTS
# =============================================================================

@data_bp.route('/data/datasets', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def create_dataset():
    """
    Create a new dataset
    """
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided for dataset creation', 'status': 'error'}), 400

        dataset_data = data['data']
        name = data.get('name', f"dataset_{uuid.uuid4().hex[:8]}")
        description = data.get('description', '')
        tags = data.get('tags', [])

        if not isinstance(dataset_data, list):
            return jsonify({'error': 'Data must be a list of records', 'status': 'error'}), 400

        # Create dataset
        dataset_id = str(uuid.uuid4())
        dataset = {
            'id': dataset_id,
            'name': name,
            'description': description,
            'tags': tags,
            'data': dataset_data,
            'record_count': len(dataset_data),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

        _mock_data_sets[dataset_id] = dataset

        telemetry_logger.log_info(f"Dataset created: {dataset_id}")

        return jsonify({
            'status': 'success',
            'message': 'Dataset created successfully',
            'dataset': {
                'id': dataset_id,
                'name': name,
                'record_count': len(dataset_data),
                'created_at': dataset['created_at']
            }
        }), 201

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'create_dataset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@data_bp.route('/data/datasets', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def list_datasets():
    """
    List all datasets
    """
    try:
        # Parse query parameters
        limit = min(int(request.args.get('limit', 50)), 100)
        offset = int(request.args.get('offset', 0))
        tags = request.args.get('tags', '').split(',') if request.args.get('tags') else []

        # Get datasets
        datasets = list(_mock_data_sets.values())

        # Filter by tags if provided
        if tags:
            datasets = [
                d for d in datasets
                if any(tag in d.get('tags', []) for tag in tags)
            ]

        # Apply pagination
        paginated_datasets = datasets[offset:offset + limit]

        # Return metadata only (not the actual data)
        dataset_list = [
            {
                'id': d['id'],
                'name': d['name'],
                'description': d['description'],
                'tags': d['tags'],
                'record_count': d['record_count'],
                'created_at': d['created_at'],
                'updated_at': d['updated_at']
            }
            for d in paginated_datasets
        ]

        return jsonify({
            'status': 'success',
            'datasets': dataset_list,
            'count': len(dataset_list),
            'total_count': len(datasets)
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'list_datasets'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@data_bp.route('/data/datasets/<dataset_id>', methods=['GET'])
@token_auth_required
@conditional_limit("20 per minute")
def get_dataset(dataset_id):
    """
    Get specific dataset
    """
    try:
        dataset = _mock_data_sets.get(dataset_id)

        if not dataset:
            return jsonify({'error': 'Dataset not found', 'status': 'error'}), 404

        return jsonify({
            'status': 'success',
            'dataset': dataset
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_dataset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@data_bp.route('/data/datasets/<dataset_id>', methods=['PUT'])
@token_auth_required
@conditional_limit("10 per minute")
def update_dataset(dataset_id):
    """
    Update dataset
    """
    try:
        dataset = _mock_data_sets.get(dataset_id)

        if not dataset:
            return jsonify({'error': 'Dataset not found', 'status': 'error'}), 404

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400

        # Update dataset metadata
        dataset.update({
            'name': data.get('name', dataset['name']),
            'description': data.get('description', dataset['description']),
            'tags': data.get('tags', dataset['tags']),
            'updated_at': datetime.now(timezone.utc).isoformat()
        })

        # Update data if provided
        if 'data' in data:
            new_data = data['data']
            if isinstance(new_data, list):
                dataset['data'] = new_data
                dataset['record_count'] = len(new_data)

        telemetry_logger.log_info(f"Dataset updated: {dataset_id}")

        return jsonify({
            'status': 'success',
            'message': 'Dataset updated successfully',
            'dataset': {
                'id': dataset_id,
                'name': dataset['name'],
                'record_count': dataset['record_count'],
                'updated_at': dataset['updated_at']
            }
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'update_dataset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


@data_bp.route('/data/datasets/<dataset_id>', methods=['DELETE'])
@token_auth_required
@conditional_limit("5 per minute")
def delete_dataset(dataset_id):
    """
    Delete dataset
    """
    try:
        if dataset_id not in _mock_data_sets:
            return jsonify({'error': 'Dataset not found', 'status': 'error'}), 404

        del _mock_data_sets[dataset_id]

        telemetry_logger.log_info(f"Dataset deleted: {dataset_id}")

        return jsonify({
            'status': 'success',
            'message': 'Dataset deleted successfully'
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'delete_dataset'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# DATA ANALYSIS ENDPOINTS
# =============================================================================

@data_bp.route('/data/analyze', methods=['POST'])
@token_auth_required
@conditional_limit("10 per minute")
def analyze_data():
    """
    Perform basic data analysis
    """
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided for analysis', 'status': 'error'}), 400

        input_data = data['data']
        analysis_type = data.get('analysis_type', 'summary')

        if not isinstance(input_data, list):
            return jsonify({'error': 'Data must be a list of records', 'status': 'error'}), 400

        # Perform analysis (mock for demo)
        analysis_result = {
            'record_count': len(input_data),
            'field_count': len(input_data[0]) if input_data else 0,
            'analysis_type': analysis_type
        }

        if analysis_type == 'summary':
            # Basic summary statistics
            numeric_fields = []
            for record in input_data[:10]:  # Sample first 10 records
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        numeric_fields.append(key)

            analysis_result['numeric_fields'] = list(set(numeric_fields))
            analysis_result['summary_stats'] = {
                'total_records': len(input_data),
                'estimated_size_mb': len(str(input_data)) / (1024 * 1024)
            }

        return jsonify({
            'status': 'success',
            'analysis': analysis_result,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'analyze_data'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500


# =============================================================================
# DATA DASHBOARD ENDPOINTS
# =============================================================================

@data_bp.route('/data/dashboard', methods=['GET'])
@token_auth_required
@conditional_limit("10 per minute")
def get_data_dashboard():
    """
    Get data management dashboard overview
    """
    try:
        datasets = list(_mock_data_sets.values())

        # Calculate dashboard stats
        total_datasets = len(datasets)
        total_records = sum(d.get('record_count', 0) for d in datasets)
        total_size_mb = sum(len(str(d.get('data', []))) / (1024 * 1024) for d in datasets)

        # Get recent datasets
        recent_datasets = sorted(
            datasets,
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )[:5]

        # Get popular tags
        all_tags = []
        for d in datasets:
            all_tags.extend(d.get('tags', []))
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        popular_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        dashboard = {
            'stats': {
                'total_datasets': total_datasets,
                'total_records': total_records,
                'total_size_mb': round(total_size_mb, 2),
                'avg_records_per_dataset': total_records // max(total_datasets, 1)
            },
            'recent_datasets': [
                {
                    'id': d['id'],
                    'name': d['name'],
                    'record_count': d['record_count'],
                    'created_at': d['created_at']
                }
                for d in recent_datasets
            ],
            'popular_tags': [{'tag': tag, 'count': count} for tag, count in popular_tags],
            'format_usage': {
                'json': total_datasets,  # Mock data
                'csv': 0,
                'xml': 0
            }
        }

        return jsonify({
            'status': 'success',
            'dashboard': dashboard
        }), 200

    except Exception as e:
        telemetry_logger.log_error(e, {'context': 'get_data_dashboard'})
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500
