"""
Data Importer Module for JPMorgan Financial APIs
Provides functionality for importing user data from various formats.
"""

import csv
import json
import io
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
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

try:
    from src.database_fixed import db_manager
except ImportError:
    db_manager = None

try:
    from src.validation import InputValidator, ValidationError
except ImportError:
    class ValidationError(Exception):
        pass
    
    class InputValidator:
        @staticmethod
        def validate_user_data(data: dict) -> bool:
            return True


# =============================================================================
# DATA VALIDATION
# =============================================================================

class DataValidationError(Exception):
    """Exception raised for data validation errors"""
    pass


class DataImporterValidator:
    """Validator for imported data"""
    
    REQUIRED_FIELDS = {
        'user': ['username', 'email'],
        'account': ['account_number', 'account_type', 'balance'],
        'transaction': ['transaction_id', 'amount', 'date'],
        'payroll': ['employee_id', 'salary', 'pay_period']
    }
    
    @staticmethod
    def validate_user_record(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a user record"""
        for field in DataImporterValidator.REQUIRED_FIELDS.get('user', []):
            if field not in data or not data[field]:
                return False, f"Missing required field: {field}"
        
        # Validate email format
        email = data.get('email', '')
        if '@' not in email or '.' not in email:
            return False, "Invalid email format"
        
        return True, None
    
    @staticmethod
    def validate_account_record(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate an account record"""
        for field in DataImporterValidator.REQUIRED_FIELDS.get('account', []):
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate account type
        valid_types = ['checking', 'savings', 'investment', 'credit', 'loan']
        account_type = data.get('account_type', '').lower()
        if account_type not in valid_types:
            return False, f"Invalid account type. Must be one of: {', '.join(valid_types)}"
        
        # Validate balance is numeric
        try:
            float(data.get('balance', 0))
        except (ValueError, TypeError):
            return False, "Balance must be a numeric value"
        
        return True, None
    
    @staticmethod
    def validate_transaction_record(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a transaction record"""
        for field in DataImporterValidator.REQUIRED_FIELDS.get('transaction', []):
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate amount is numeric
        try:
            float(data.get('amount', 0))
        except (ValueError, TypeError):
            return False, "Amount must be a numeric value"
        
        return True, None
    
    @staticmethod
    def validate_payroll_record(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a payroll record"""
        for field in DataImporterValidator.REQUIRED_FIELDS.get('payroll', []):
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate salary is numeric
        try:
            float(data.get('salary', 0))
        except (ValueError, TypeError):
            return False, "Salary must be a numeric value"
        
        return True, None


# =============================================================================
# DATA IMPORTER
# =============================================================================

class DataImporter:
    """Importer for user data from various formats"""
    
    def __init__(self):
        self.logger = telemetry_logger
        self.validator = DataImporterValidator()
        self.imported_records = []
        self.failed_records = []
        self.errors = []
    
    def import_from_json(self, json_data: str, data_type: str = 'user') -> Dict[str, Any]:
        """
        Import data from JSON string.
        
        Args:
            json_data: JSON string containing the data
            data_type: Type of data ('user', 'account', 'transaction', 'payroll')
            
        Returns:
            Dict with import results
        """
        self.imported_records = []
        self.failed_records = []
        self.errors = []
        
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            return {
                'status': 'error',
                'message': f'Invalid JSON format: {str(e)}',
                'imported_count': 0,
                'failed_count': 0
            }
        
        # Handle both single record and array of records
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            return {
                'status': 'error',
                'message': 'Invalid data format. Expected dict or list.',
                'imported_count': 0,
                'failed_count': 0
            }
        
        return self._process_records(records, data_type)
    
    def import_from_csv(self, csv_data: str, data_type: str = 'user') -> Dict[str, Any]:
        """
        Import data from CSV string.
        
        Args:
            csv_data: CSV string containing the data
            data_type: Type of data ('user', 'account', 'transaction', 'payroll')
            
        Returns:
            Dict with import results
        """
        self.imported_records = []
        self.failed_records = []
        self.errors = []
        
        try:
            # Parse CSV data
            reader = csv.DictReader(io.StringIO(csv_data))
            records = list(reader)
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Invalid CSV format: {str(e)}',
                'imported_count': 0,
                'failed_count': 0
            }
        
        return self._process_records(records, data_type)
    
    def import_from_dict_list(self, records: List[Dict[str, Any]], data_type: str = 'user') -> Dict[str, Any]:
        """
        Import data from a list of dictionaries.
        
        Args:
            records: List of dictionaries containing the data
            data_type: Type of data ('user', 'account', 'transaction', 'payroll')
            
        Returns:
            Dict with import results
        """
        self.imported_records = []
        self.failed_records = []
        self.errors = []
        
        return self._process_records(records, data_type)
    
    def _process_records(self, records: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """Process a list of records"""
        
        validation_method = getattr(self.validator, f'validate_{data_type}_record', None)
        
        for idx, record in enumerate(records):
            try:
                # Validate record
                if validation_method:
                    is_valid, error_message = validation_method(record)
                    if not is_valid:
                        self.failed_records.append({
                            'record': record,
                            'error': error_message,
                            'row': idx + 1
                        })
                        self.errors.append(f"Row {idx + 1}: {error_message}")
                        continue
                
                # Process record
                processed_record = self._process_record(record, data_type)
                self.imported_records.append(processed_record)
                
            except Exception as e:
                self.failed_records.append({
                    'record': record,
                    'error': str(e),
                    'row': idx + 1
                })
                self.errors.append(f"Row {idx + 1}: {str(e)}")
        
        # Save to database if applicable
        saved_count = 0
        if db_manager and self.imported_records:
            saved_count = self._save_to_database(data_type)
        
        self.logger.log_info(
            f"Data import completed: {len(self.imported_records)} imported, {len(self.failed_records)} failed",
            {'context': 'data_importer', 'data_type': data_type}
        )
        
        return {
            'status': 'success' if self.imported_records else 'error',
            'message': f"Imported {len(self.imported_records)} records, {len(self.failed_records)} failed",
            'imported_count': len(self.imported_records),
            'failed_count': len(self.failed_records),
            'saved_count': saved_count,
            'errors': self.errors[:10] if self.errors else [],  # Return first 10 errors
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _process_record(self, record: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Process a single record"""
        
        # Add metadata
        processed = record.copy()
        processed['_imported_at'] = datetime.now(timezone.utc).isoformat()
        processed['_data_type'] = data_type
        
        # Convert string dates to ISO format
        date_fields = ['date', 'created_at', 'updated_at', 'dob', 'transaction_date']
        for field in date_fields:
            if field in processed and isinstance(processed[field], str):
                try:
                    # Try to parse and reformat date
                    dt = datetime.fromisoformat(processed[field].replace('Z', '+00:00'))
                    processed[field] = dt.isoformat()
                except (ValueError, AttributeError):
                    pass
        
        return processed
    
    def _save_to_database(self, data_type: str) -> int:
        """Save imported records to database"""
        
        saved_count = 0
        
        try:
            if data_type == 'account' and db_manager:
                for record in self.imported_records:
                    try:
                        business_data = {
                            'name': record.get('account_name', record.get('username', 'Unknown')),
                            'type': record.get('account_type', 'checking'),
                            'registration_number': record.get('account_number', ''),
                            'address': record.get('address', ''),
                            'contact_info': json.dumps({
                                'email': record.get('email', ''),
                                'phone': record.get('phone', '')
                            })
                        }
                        db_manager.create_business(business_data)
                        saved_count += 1
                    except Exception as e:
                        self.errors.append(f"Database save error: {str(e)}")
            
        except Exception as e:
            self.logger.log_error(e, {'context': 'data_importer_save'})
        
        return saved_count
    
    def get_import_summary(self) -> Dict[str, Any]:
        """Get summary of last import operation"""
        return {
            'total_processed': len(self.imported_records) + len(self.failed_records),
            'imported_count': len(self.imported_records),
            'failed_count': len(self.failed_records),
            'success_rate': (len(self.imported_records) / (len(self.imported_records) + len(self.failed_records)) * 100) 
                          if (len(self.imported_records) + len(self.failed_records)) > 0 else 0,
            'errors': self.errors
        }


# =============================================================================
# BATCH IMPORTER
# =============================================================================

class BatchImporter:
    """Batch importer for large datasets"""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.importer = DataImporter()
    
    def import_large_dataset(self, records: List[Dict[str, Any]], data_type: str = 'user') -> Dict[str, Any]:
        """
        Import a large dataset in batches.
        
        Args:
            records: List of records to import
            data_type: Type of data
            
        Returns:
            Dict with overall import results
        """
        total_imported = 0
        total_failed = 0
        all_errors = []
        
        # Process in batches
        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]
            result = self.importer.import_from_dict_list(batch, data_type)
            
            total_imported += result.get('imported_count', 0)
            total_failed += result.get('failed_count', 0)
            all_errors.extend(result.get('errors', []))
        
        return {
            'status': 'success' if total_imported > 0 else 'error',
            'message': f"Batch import completed: {total_imported} imported, {total_failed} failed",
            'total_imported': total_imported,
            'total_failed': total_failed,
            'batches_processed': (len(records) + self.batch_size - 1) // self.batch_size,
            'errors': all_errors[:20],  # Return first 20 errors
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DataImporter',
    'BatchImporter',
    'DataImporterValidator',
    'DataValidationError'
]
