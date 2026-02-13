"""
Payroll Blueprint for JPMorgan Financial APIs
Provides endpoints for payroll management.
"""

import os
from flask import Blueprint, request, jsonify, g
from functools import wraps
from datetime import datetime, timezone

try:
    from src.auth import token_auth_required, require_auth
except ImportError:
    # Fallback if auth module not available
    def token_auth_required(f):
        return f
    def require_auth(f):
        return f

try:
    from src.payroll_service import payroll_service
except ImportError:
    from src.payroll_service import PayrollService
    payroll_service = PayrollService()

try:
    from src.logger import telemetry_logger
except ImportError:
    class FallbackLogger:
        def log_info(self, msg, context=None):
            print(f"INFO: {msg}")
        def log_error(self, msg, context=None):
            print(f"ERROR: {msg}")
    telemetry_logger = FallbackLogger()


# Create blueprint
payroll_bp = Blueprint('payroll', __name__)


# =============================================================================
# EMPLOYEE ENDPOINTS
# =============================================================================

@payroll_bp.route('/employees', methods=['POST'])
@token_auth_required
def create_employee():
    """
    Create a new employee
    ---
    Tags:
      - Payroll
    Parameters:
        - in: body
          name: body
          required: true
          schema:
            type: object
            required:
              - first_name
              - last_name
              - email
            properties:
              first_name:
                type: string
                description: Employee's first name
              last_name:
                type: string
                description: Employee's last name
              email:
                type: string
                description: Employee's email
              phone:
                type: string
                description: Employee's phone number
              department:
                type: string
                description: Department
              position:
                type: string
                description: Position/Job title
              salary:
                type: number
                description: Annual salary
              hourly_rate:
                type: number
                description: Hourly rate (for hourly employees)
              pay_frequency:
                type: string
                enum: [weekly, biweekly, monthly]
                description: Pay frequency
    responses:
      201:
        description: Employee created successfully
      400:
        description: Invalid request
    """
    try:
        data = request.get_json()
        user_id = g.get('user_id', 'test_user')
        
        result = payroll_service.create_employee(user_id, data)
        
        if result['status'] == 'error':
            return jsonify(result), 400
        
        telemetry_logger.log_info(f"Employee created: {result['employee']['employee_id']}", {'context': 'payroll'})
        
        return jsonify(result), 201
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'create_employee'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@payroll_bp.route('/employees', methods=['GET'])
@token_auth_required
def list_employees():
    """
    List all employees
    ---
    Tags:
      - Payroll
    responses:
      200:
        description: List of employees
    """
    try:
        user_id = g.get('user_id', 'test_user')
        
        result = payroll_service.list_employees(user_id)
        
        return jsonify(result), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'list_employees'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@payroll_bp.route('/employees/<employee_id>', methods=['GET'])
@token_auth_required
def get_employee(employee_id):
    """
    Get an employee by ID
    ---
    Tags:
      - Payroll
    Parameters:
        - in: path
          name: employee_id
          required: true
          type: string
    responses:
      200:
        description: Employee details
      404:
        description: Employee not found
    """
    try:
        result = payroll_service.get_employee(employee_id)
        
        if result['status'] == 'error':
            return jsonify(result), 404
        
        return jsonify(result), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'get_employee'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@payroll_bp.route('/employees/<employee_id>', methods=['PUT'])
@token_auth_required
def update_employee(employee_id):
    """
    Update an employee
    ---
    Tags:
      - Payroll
    Parameters:
        - in: path
          name: employee_id
          required: true
          type: string
        - in: body
          name: body
          required: true
          schema:
            type: object
            properties:
              first_name:
                type: string
              last_name:
                type: string
              email:
                type: string
              phone:
                type: string
              department:
                type: string
              position:
                type: string
              salary:
                type: number
              hourly_rate:
                type: number
              pay_frequency:
                type: string
                enum: [weekly, biweekly, monthly]
              status:
                type: string
                enum: [active, inactive, terminated]
    responses:
      200:
        description: Employee updated successfully
      404:
        description: Employee not found
    """
    try:
        data = request.get_json()
        
        result = payroll_service.update_employee(employee_id, data)
        
        if result['status'] == 'error':
            return jsonify(result), 404
        
        telemetry_logger.log_info(f"Employee updated: {employee_id}", {'context': 'payroll'})
        
        return jsonify(result), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'update_employee'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@payroll_bp.route('/employees/<employee_id>', methods=['DELETE'])
@token_auth_required
def delete_employee(employee_id):
    """
    Delete an employee
    ---
    Tags:
      - Payroll
    Parameters:
        - in: path
          name: employee_id
          required: true
          type: string
    responses:
      200:
        description: Employee deleted successfully
      404:
        description: Employee not found
    """
    try:
        result = payroll_service.delete_employee(employee_id)
        
        if result['status'] == 'error':
            return jsonify(result), 404
        
        telemetry_logger.log_info(f"Employee deleted: {employee_id}", {'context': 'payroll'})
        
        return jsonify(result), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'delete_employee'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =============================================================================
# PAYROLL RUN ENDPOINTS
# =============================================================================

@payroll_bp.route('/runs', methods=['POST'])
@token_auth_required
def create_payroll_run():
    """
    Create a new payroll run
    ---
    Tags:
      - Payroll
    Parameters:
        - in: body
          name: body
          required: true
          schema:
            type: object
            required:
              - pay_period_start
              - pay_period_end
              - payment_date
            properties:
              pay_period_start:
                type: string
                format: date
                description: Start of pay period
              pay_period_end:
                type: string
                format: date
                description: End of pay period
              payment_date:
                type: string
                format: date
                description: Date of payment
    responses:
      201:
        description: Payroll run created successfully
      400:
        description: Invalid request
    """
    try:
        data = request.get_json()
        user_id = g.get('user_id', 'test_user')
        
        result = payroll_service.create_payroll_run(user_id, data)
        
        if result['status'] == 'error':
            return jsonify(result), 400
        
        telemetry_logger.log_info(f"Payroll run created: {result['run']['run_id']}", {'context': 'payroll'})
        
        return jsonify(result), 201
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'create_payroll_run'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@payroll_bp.route('/runs', methods=['GET'])
@token_auth_required
def list_payroll_runs():
    """
    List all payroll runs
    ---
    Tags:
      - Payroll
    responses:
      200:
        description: List of payroll runs
    """
    try:
        user_id = g.get('user_id', 'test_user')
        
        result = payroll_service.list_payroll_runs(user_id)
        
        return jsonify(result), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'list_payroll_runs'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@payroll_bp.route('/runs/<run_id>', methods=['GET'])
@token_auth_required
def get_payroll_run(run_id):
    """
    Get a payroll run by ID
    ---
    Tags:
      - Payroll
    Parameters:
        - in: path
          name: run_id
          required: true
          type: string
    responses:
      200:
        description: Payroll run details
      404:
        description: Payroll run not found
    """
    try:
        result = payroll_service.get_payroll_run(run_id)
        
        if result['status'] == 'error':
            return jsonify(result), 404
        
        return jsonify(result), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'get_payroll_run'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@payroll_bp.route('/runs/<run_id>/process', methods=['POST'])
@token_auth_required
def process_payroll_run(run_id):
    """
    Process a payroll run
    ---
    Tags:
      - Payroll
    Parameters:
        - in: path
          name: run_id
          required: true
          type: string
    responses:
      200:
        description: Payroll run processed successfully
      400:
        description: Invalid request
      404:
        description: Payroll run not found
    """
    try:
        result = payroll_service.process_payroll_run(run_id)
        
        if result['status'] == 'error':
            return jsonify(result), 400 if 'not found' in result.get('message', '').lower() else 400
        
        telemetry_logger.log_info(f"Payroll run processed: {run_id}", {'context': 'payroll'})
        
        return jsonify(result), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'process_payroll_run'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =============================================================================
# PAYMENT ENDPOINTS
# =============================================================================

@payroll_bp.route('/payments/<payment_id>', methods=['GET'])
@token_auth_required
def get_payment(payment_id):
    """
    Get a payment by ID
    ---
    Tags:
      - Payroll
    Parameters:
        - in: path
          name: payment_id
          required: true
          type: string
    responses:
      200:
        description: Payment details
      404:
        description: Payment not found
    """
    try:
        result = payroll_service.get_payment(payment_id)
        
        if result['status'] == 'error':
            return jsonify(result), 404
        
        return jsonify(result), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'get_payment'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


@payroll_bp.route('/employees/<employee_id>/payments', methods=['GET'])
@token_auth_required
def get_employee_payments(employee_id):
    """
    Get all payments for an employee
    ---
    Tags:
      - Payroll
    Parameters:
        - in: path
          name: employee_id
          required: true
          type: string
    responses:
      200:
        description: List of payments
    """
    try:
        result = payroll_service.get_employee_payments(employee_id)
        
        return jsonify(result), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'get_employee_payments'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =============================================================================
# TAX CALCULATION ENDPOINTS
# =============================================================================

@payroll_bp.route('/employees/<employee_id>/taxes', methods=['GET'])
@token_auth_required
def calculate_employee_taxes(employee_id):
    """
    Calculate estimated taxes for an employee
    ---
    Tags:
      - Payroll
    Parameters:
        - in: path
          name: employee_id
          required: true
          type: string
    responses:
      200:
        description: Tax calculation details
      404:
        description: Employee not found
    """
    try:
        result = payroll_service.calculate_employee_taxes(employee_id)
        
        if result['status'] == 'error':
            return jsonify(result), 404
        
        return jsonify(result), 200
    
    except Exception as e:
        telemetry_logger.log_error(str(e), {'context': 'calculate_employee_taxes'})
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =============================================================================
# HEALTH CHECK
# =============================================================================

@payroll_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check for payroll service
    ---
    Tags:
      - Payroll
    responses:
      200:
        description: Service is healthy
    """
    return jsonify({
        'status': 'healthy',
        'service': 'payroll',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }), 200


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['payroll_bp']
