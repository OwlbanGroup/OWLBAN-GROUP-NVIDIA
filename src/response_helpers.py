"""Response helper functions"""
from flask import jsonify
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

def success_response(data: Dict[str, Any], status_code: int = 200) -> Tuple:
    """Standardized success response"""
    response = {
        'status': 'success',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    response.update(data)
    return jsonify(response), status_code

def error_response(message: str, status_code: int = 500, error_code: str = None) -> Tuple:
    """Standardized error response"""
    response = {
        'status': 'error',
        'error': message,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    if error_code:
        response['error_code'] = error_code
    return jsonify(response), status_code

def validation_error_response(errors: Dict[str, str]) -> Tuple:
    """Validation error response"""
    return jsonify({
        'status': 'error',
        'error': 'Validation failed',
        'validation_errors': errors,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 400
