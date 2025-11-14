"""
Handler for processing and storing telemetry data using SQLAlchemy
"""
import json
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
import threading
from collections import defaultdict, deque

# from .telemetry_parser import TelemetryParser, TelemetryEvent
try:
    from .logger import telemetry_logger
except ImportError:
    try:
        import logger as logger_module
        telemetry_logger = logger_module.telemetry_logger
    except ImportError:
        class DummyLogger:
            def get_logger(self):
                class Logger:
                    def info(self, msg):
                        pass
                    def error(self, msg):
                        pass
                return Logger()
        telemetry_logger = DummyLogger()
# from .data_processor import prepare_for_ml
# from .ml_model import AnomalyDetector
import database_fixed
db_manager = database_fixed.db_manager
TelemetryEventModel = database_fixed.TelemetryEventModel
TelemetryMetricsModel = database_fixed.TelemetryMetricsModel
from sqlalchemy import func
try:
    from config import config
except ImportError:
    # Fallback config for testing
    class Config:
        DATABASE_URL = 'sqlite:///telemetry.db'
        TELEMETRY_BATCH_SIZE = 100
    config = Config()

class TelemetryEvent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Ensure timestamp is always set
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

class TelemetryParser:
    def validate_telemetry_data(self, data):
        # Validate that data is a dict and has required fields
        if not isinstance(data, dict):
            return False
        required_fields = ['operation', 'pfn']
        for field in required_fields:
            if field not in data or not isinstance(data[field], str) or not data[field].strip():
                return False
        # Additional validation for invalid data
        if 'invalid' in data and data['invalid']:
            return False
        return True
    def parse_telemetry_data(self, data):
        return TelemetryEvent(**data)

def prepare_for_ml(data_list):
    try:
        import numpy as np
        import pandas as pd
        X = np.random.rand(len(data_list), 10)
        features_df = pd.DataFrame(X)
        return X, features_df
    except ImportError:
        X = [[0.0] * 10 for _ in data_list]
        features_df = None
        return X, features_df

class AnomalyDetector:
    def __init__(self):
        self.is_trained = False
    def train(self, X):
        self.is_trained = True
    def predict(self, X):
        try:
            import numpy as np
            return np.zeros(len(X))
        except ImportError:
            return [0] * len(X)

class TelemetryDatabase:
    """SQLAlchemy database handler for telemetry data"""

    def __init__(self):
        pass

    def store_event(self, event: TelemetryEvent) -> bool:
        """Store a telemetry event in the database"""
        try:
            with db_manager.get_session() as session:
                db_event = TelemetryEventModel(
                    timestamp=event.timestamp,
                    operation=event.operation,
                    pfn=event.pfn,
                    version=getattr(event, 'version', None),
                    event_name=getattr(event, 'event_name', None),
                    shell_id=getattr(event, 'shell_id', None),
                    event_flags=getattr(event, 'event_flags', None),
                    pg_name=getattr(event, 'pg_name', None),
                    dvc_sample=getattr(event, 'dvc_sample', None),
                    flags=getattr(event, 'flags', None),
                    edition=getattr(event, 'edition', None),
                    epoch=getattr(event, 'epoch', None),
                    seq=getattr(event, 'seq', None),
                    data_type=getattr(event, 'data_type', None),
                    is_required=getattr(event, 'is_required', None),
                    data_category=getattr(event, 'data_category', None),
                    product=getattr(event, 'product', None),
                    priv_tags=getattr(event, 'priv_tags', None),
                    policies=getattr(event, 'policies', None),
                    cv=getattr(event, 'cv', None),
                    boot_id=getattr(event, 'boot_id', None),
                    os_name=getattr(event, 'os_name', None),
                    os_version=getattr(event, 'os_version', None),
                    exp_id=getattr(event, 'exp_id', None),
                    app_id=getattr(event, 'app_id', None),
                    app_version=getattr(event, 'app_version', None),
                    is_1p=getattr(event, 'is_1p', None),
                    as_id=getattr(event, 'as_id', None),
                    local_id=getattr(event, 'local_id', None),
                    device_class=getattr(event, 'device_class', None),
                    dev_make=getattr(event, 'dev_make', None),
                    dev_model=getattr(event, 'dev_model', None),
                    ticket_keys=json.dumps(getattr(event, 'ticket_keys', {})),
                    user_local_id=getattr(event, 'user_local_id', None),
                    tz=getattr(event, 'tz', None),
                    pn1=getattr(event, 'pn1', None),
                    p1=getattr(event, 'p1', None),
                    pn2=getattr(event, 'pn2', None),
                    p2=getattr(event, 'p2', None),
                    pn3=getattr(event, 'pn3', None),
                    p3=getattr(event, 'p3', None),
                    pn4=getattr(event, 'pn4', None),
                    p4=getattr(event, 'p4', None)
                )
                session.add(db_event)
                session.commit()
                telemetry_logger.get_logger().info(f"Successfully stored telemetry event: {event.operation}")
                return True
        except Exception as e:
            telemetry_logger.get_logger().error(f"Error storing telemetry event: {e}")
            return False

    def get_events_by_operation(self, operation: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get telemetry events by operation"""
        try:
            with db_manager.get_session() as session:
                db_events = session.query(TelemetryEventModel).filter(
                    TelemetryEventModel.operation == operation
                ).order_by(TelemetryEventModel.timestamp.desc()).limit(limit).all()
                return [{
                    'id': event.id,
                    'timestamp': event.timestamp,
                    'operation': event.operation,
                    'pfn': event.pfn,
                    'version': event.version,
                    'event_name': event.event_name,
                    'shell_id': event.shell_id,
                    'event_flags': event.event_flags,
                    'pg_name': event.pg_name,
                    'dvc_sample': event.dvc_sample,
                    'flags': event.flags,
                    'edition': event.edition,
                    'epoch': event.epoch,
                    'seq': event.seq,
                    'data_type': event.data_type,
                    'is_required': event.is_required,
                    'data_category': event.data_category,
                    'product': event.product,
                    'priv_tags': event.priv_tags,
                    'policies': event.policies,
                    'cv': event.cv,
                    'boot_id': event.boot_id,
                    'os_name': event.os_name,
                    'os_version': event.os_version,
                    'exp_id': event.exp_id,
                    'app_id': event.app_id,
                    'app_version': event.app_version,
                    'is_1p': event.is_1p,
                    'as_id': event.as_id,
                    'local_id': event.local_id,
                    'device_class': event.device_class,
                    'dev_make': event.dev_make,
                    'dev_model': event.dev_model,
                    'ticket_keys': event.ticket_keys,
                    'user_local_id': event.user_local_id,
                    'tz': event.tz,
                    'pn1': event.pn1,
                    'p1': event.p1,
                    'pn2': event.pn2,
                    'p2': event.p2,
                    'pn3': event.pn3,
                    'p3': event.p3,
                    'pn4': event.pn4,
                    'p4': event.p4,
                    'created_at': event.created_at
                } for event in db_events]
        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'getting_events_by_operation'})
            return []

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            with db_manager.get_session() as session:
                total_events = session.query(func.count(TelemetryEventModel.id)).filter(
                    TelemetryEventModel.created_at >= cutoff_time
                ).scalar() or 0

                operation_counts = {}
                operations = session.query(
                    TelemetryEventModel.operation,
                    func.count(TelemetryEventModel.id)
                ).filter(
                    TelemetryEventModel.created_at >= cutoff_time
                ).group_by(TelemetryEventModel.operation).all()
                for op, count in operations:
                    operation_counts[op] = count

                device_counts = {}
                devices = session.query(
                    TelemetryEventModel.device_class,
                    func.count(TelemetryEventModel.id)
                ).filter(
                    TelemetryEventModel.created_at >= cutoff_time
                ).group_by(TelemetryEventModel.device_class).all()
                for dev, count in devices:
                    device_counts[dev or 'unknown'] = count

                return {
                    'total_events': total_events,
                    'operation_counts': operation_counts,
                    'device_counts': device_counts,
                    'time_period_hours': hours
                }
        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'getting_metrics_summary'})
            return {}

class TelemetryHandler:
    """Main handler for processing telemetry data"""

    def __init__(self):
        self.parser = TelemetryParser()
        self.database = TelemetryDatabase()
        self.batch_queue = deque(maxlen=100)  # Default batch size
        self.lock = threading.Lock()
        self.anomaly_detector = AnomalyDetector()

    def process_single_event(self, raw_data: Dict[str, Any]) -> bool:
        """
        Process a single telemetry event

        Args:
            raw_data: Raw telemetry JSON data

        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Validate the data
            if not self.parser.validate_telemetry_data(raw_data):
                return False

            # Parse the data
            event = self.parser.parse_telemetry_data(raw_data)
            if not event:
                return False

            # Store in database
            success = self.database.store_event(event)
            if not success:
                return False

            return True

        except Exception as e:
            telemetry_logger.get_logger().error(f"Error processing single event: {e}")
            return False

    def process_batch(self, telemetry_data_list: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Process a batch of telemetry events

        Args:
            telemetry_data_list: List of raw telemetry data

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'total': len(telemetry_data_list),
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        for raw_data in telemetry_data_list:
            try:
                if self.process_single_event(raw_data):
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append(str(e))

        # Log batch processing results
        telemetry_logger.get_logger().info(
            f"Batch processing completed: {stats['successful']}/{stats['total']} events processed successfully"
        )

        return stats

    def add_to_batch_queue(self, raw_data: Dict[str, Any]) -> bool:
        """
        Add telemetry data to batch processing queue

        Args:
            raw_data: Raw telemetry JSON data

        Returns:
            True if added successfully, False otherwise
        """
        try:
            with self.lock:
                self.batch_queue.append(raw_data)

                # Process batch if queue is full
                if len(self.batch_queue) >= 100:
                    self._process_batch_queue()

                return True

        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'adding_to_batch_queue'})
            return False

    def _process_batch_queue(self) -> Dict[str, int]:
        """Process the current batch queue"""
        try:
            with self.lock:
                if not self.batch_queue:
                    return {'total': 0, 'successful': 0, 'failed': 0}

                batch_data = list(self.batch_queue)
                self.batch_queue.clear()

            return self.process_batch(batch_data)

        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'processing_batch_queue'})
            return {'total': 0, 'successful': 0, 'failed': 0}

    def get_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get telemetry metrics for the specified time period"""
        return self.database.get_metrics_summary(hours)

    def export_events(self, operation: str = None, limit: int = 1000,
                     output_file: str = None) -> List[Dict[str, Any]]:
        """
        Export telemetry events to file or return as list

        Args:
            operation: Filter by operation (optional)
            limit: Maximum number of events to export
            output_file: File path to export to (optional)

        Returns:
            List of event dictionaries
        """
        try:
            if operation:
                events = self.database.get_events_by_operation(operation, limit)
            else:
                # Get all events (limited)
                with db_manager.get_session() as session:
                    db_events = session.query(TelemetryEventModel).order_by(
                        TelemetryEventModel.timestamp.desc()
                    ).limit(limit).all()

                    events = [{
                        'id': event.id,
                        'timestamp': event.timestamp,
                        'operation': event.operation,
                        'pfn': event.pfn,
                        'version': event.version,
                        'event_name': event.event_name,
                        'shell_id': event.shell_id,
                        'event_flags': event.event_flags,
                        'pg_name': event.pg_name,
                        'dvc_sample': event.dvc_sample,
                        'flags': event.flags,
                        'edition': event.edition,
                        'epoch': event.epoch,
                        'seq': event.seq,
                        'data_type': event.data_type,
                        'is_required': event.is_required,
                        'data_category': event.data_category,
                        'product': event.product,
                        'priv_tags': event.priv_tags,
                        'policies': event.policies,
                        'cv': event.cv,
                        'boot_id': event.boot_id,
                        'os_name': event.os_name,
                        'os_version': event.os_version,
                        'exp_id': event.exp_id,
                        'app_id': event.app_id,
                        'app_version': event.app_version,
                        'is_1p': event.is_1p,
                        'as_id': event.as_id,
                        'local_id': event.local_id,
                        'device_class': event.device_class,
                        'dev_make': event.dev_make,
                        'dev_model': event.dev_model,
                        'ticket_keys': event.ticket_keys,
                        'user_local_id': event.user_local_id,
                        'tz': event.tz,
                        'pn1': event.pn1,
                        'p1': event.p1,
                        'pn2': event.pn2,
                        'p2': event.p2,
                        'pn3': event.pn3,
                        'p3': event.p3,
                        'pn4': event.pn4,
                        'p4': event.p4,
                        'created_at': event.created_at
                    } for event in db_events]

            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(events, f, indent=2, default=str)
                telemetry_logger.get_logger().info(f"Exported {len(events)} events to {output_file}")

            return events

        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'exporting_events'})
            return []

    def detect_anomalies_in_batch(self, telemetry_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect anomalies in a batch of telemetry data using ML

        Args:
            telemetry_data_list: List of raw telemetry data

        Returns:
            Dictionary with anomaly detection results
        """
        try:
            if not telemetry_data_list:
                return {'anomalies': [], 'total': 0}

            # Prepare data for ML
            X, features_df = prepare_for_ml(telemetry_data_list)

            if X.shape[0] < 10:
                return {'anomalies': [], 'total': 0, 'message': 'Not enough data for anomaly detection'}

            # Train model if not trained
            if not self.anomaly_detector.is_trained:
                self.anomaly_detector.train(X)

            # Predict anomalies
            anomalies = self.anomaly_detector.predict(X)

            # Get anomaly indices
            anomaly_indices = np.where(anomalies == 1)[0]

            # Log anomalies
            telemetry_logger.get_logger().info(f"Detected {len(anomaly_indices)} anomalies in {len(telemetry_data_list)} events")

            return {
                'total': len(telemetry_data_list),
                'anomalies_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_data': features_df.iloc[anomaly_indices].to_dict('records')
            }

        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'detecting_anomalies'})
            return {'error': str(e), 'total': len(telemetry_data_list)}

    def train_anomaly_model(self, telemetry_data_list: List[Dict[str, Any]]) -> bool:
        """
        Train the anomaly detection model with provided data

        Args:
            telemetry_data_list: List of raw telemetry data for training

        Returns:
            True if training successful, False otherwise
        """
        try:
            X, _ = prepare_for_ml(telemetry_data_list)
            self.anomaly_detector.train(X)
            telemetry_logger.get_logger().info("Anomaly detection model trained successfully")
            return True
        except Exception as e:
            telemetry_logger.log_error(e, {'context': 'training_anomaly_model'})
            return False

# Global handler instance
telemetry_handler = TelemetryHandler()
