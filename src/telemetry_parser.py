"""
Parser for Microsoft Windows Store telemetry data
"""
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from .logger import telemetry_logger

@dataclass
class TelemetryEvent:
    """Data class representing a parsed telemetry event"""
    timestamp: str
    operation: str
    pfn: str
    version: str
    event_name: str
    shell_id: int
    event_flags: int
    pg_name: str
    dvc_sample: float
    flags: int
    edition: int
    epoch: str
    seq: int
    data_type: int
    is_required: bool
    data_category: int
    product: int
    priv_tags: int
    policies: int
    cv: str
    boot_id: int
    os_name: str
    os_version: str
    exp_id: str
    app_id: str
    app_version: str
    is_1p: int
    as_id: int
    local_id: str
    device_class: str
    dev_make: str
    dev_model: str
    ticket_keys: List[str]
    user_local_id: str
    tz: str
    pn1: str
    p1: str
    pn2: str
    p2: str
    pn3: str
    p3: str
    pn4: str
    p4: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

class TelemetryParser:
    """Parser for Microsoft Windows Store telemetry data"""

    def __init__(self):
        self.logger = telemetry_logger

    def parse_telemetry_data(self, raw_data: Dict[str, Any]) -> Optional[TelemetryEvent]:
        """
        Parse raw telemetry data into a structured TelemetryEvent

        Args:
            raw_data: Raw telemetry JSON data

        Returns:
            TelemetryEvent object if parsing successful, None otherwise
        """
        try:
            # Extract basic information
            timestamp = raw_data.get('time', '')
            operation = raw_data.get('data', {}).get('Op', '')
            pfn = raw_data.get('data', {}).get('PFN', '')
            version = raw_data.get('ver', '')

            # Extract extended information
            ext = raw_data.get('ext', {})

            # UTC extension
            utc_raw = ext.get('utc')
            utc = utc_raw if isinstance(utc_raw, dict) else {}
            shell_id = utc.get('shellId', 0)
            event_flags = utc.get('eventFlags', 0)
            pg_name = utc.get('pgName', '')
            dvc_sample = utc.get('dvcSample', 0.0)
            flags = ext.get('flags', 0)  # flags is top-level in sample
            edition = utc.get('edition', 0)
            epoch = utc.get('epoch', '')
            seq = utc.get('seq', 0)

            # Privacy extension
            privacy_raw = ext.get('privacy')
            privacy = privacy_raw if isinstance(privacy_raw, dict) else {}
            data_type = privacy.get('dataType', 0)
            is_required = privacy.get('isRequired', False)
            data_category = privacy.get('dataCategory', 0)
            product = privacy.get('product', 0)

            # Metadata extension
            metadata_raw = ext.get('metadata')
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            priv_tags = metadata.get('privTags', 0)
            policies = metadata.get('policies', 0)

            # MSCV extension
            mscv_raw = ext.get('mscv')
            mscv = mscv_raw if isinstance(mscv_raw, dict) else {}
            cv = mscv.get('cV', '')

            # OS extension
            os_raw = ext.get('os')
            os_info = os_raw if isinstance(os_raw, dict) else {}
            boot_id = os_info.get('bootId', 0)
            os_name = os_info.get('name', ext.get('OS', ''))  # Fallback to data.OS
            os_version = os_info.get('ver', '')

            # App extension
            app_raw = ext.get('app')
            app_info = app_raw if isinstance(app_raw, dict) else {}
            app_id = app_info.get('id', '')
            app_version = app_info.get('ver', '')

            # Device extension
            device_raw = ext.get('device')
            device_info = device_raw if isinstance(device_raw, dict) else {}
            local_id = device_info.get('localId', '')
            device_class = device_info.get('deviceClass', ext.get('DeviceClass', ''))  # Fallback

            # Protocol extension
            protocol_raw = ext.get('protocol')
            protocol_info = protocol_raw if isinstance(protocol_raw, dict) else {}
            dev_make = protocol_info.get('devMake', '')
            dev_model = protocol_info.get('devModel', ext.get('DeviceModel', ''))  # Fallback to data.DeviceModel
            ticket_keys = protocol_info.get('ticketKeys', [])

            # User extension
            user_raw = ext.get('user')
            user_info = user_raw if isinstance(user_raw, dict) else {}
            user_local_id = user_info.get('localId', ext.get('UserId', ''))  # Fallback to data.UserId

            # Location extension
            loc_raw = ext.get('loc')
            loc_info = loc_raw if isinstance(loc_raw, dict) else {}
            tz = loc_info.get('tz', '')

            # Parameter data
            data_info = raw_data.get('data', {})
            pn1 = data_info.get('PN1', '')
            p1 = data_info.get('P1', '')
            pn2 = data_info.get('PN2', '')
            p2 = data_info.get('P2', '')
            pn3 = data_info.get('PN3', '')
            p3 = data_info.get('P3', '')
            pn4 = data_info.get('PN4', '')
            p4 = data_info.get('P4', '')

            exp_id = os_info.get('expId', '')

            is_1p = app_info.get('is1P', 0)
            as_id = app_info.get('asId', 0)

            # Create telemetry event
            event = TelemetryEvent(
                timestamp=timestamp,
                operation=operation,
                pfn=pfn,
                version=version,
                event_name=raw_data.get('name', ''),
                shell_id=shell_id,
                event_flags=event_flags,
                pg_name=pg_name,
                dvc_sample=dvc_sample,
                flags=flags,
                edition=edition,
                epoch=epoch,
                seq=seq,
                data_type=data_type,
                is_required=is_required,
                data_category=data_category,
                product=product,
                priv_tags=priv_tags,
                policies=policies,
                cv=cv,
                boot_id=boot_id,
                os_name=os_name,
                os_version=os_version,
                exp_id=exp_id,
                app_id=app_id,
                app_version=app_version,
                is_1p=is_1p,
                as_id=as_id,
                local_id=local_id,
                device_class=device_class,
                dev_make=dev_make,
                dev_model=dev_model,
                ticket_keys=ticket_keys,
                user_local_id=user_local_id,
                tz=tz,
                pn1=pn1,
                p1=p1,
                pn2=pn2,
                p2=p2,
                pn3=pn3,
                p3=p3,
                pn4=pn4,
                p4=p4
            )

            # Log the parsed event
            self.logger.log_telemetry(raw_data, 'INFO')

            return event

        except Exception as e:
            self.logger.log_error(e, {'context': 'parsing_telemetry_data'})
            return None

    def validate_telemetry_data(self, raw_data: Dict[str, Any]) -> bool:
        """
        Validate that the telemetry data has required fields

        Args:
            raw_data: Raw telemetry JSON data

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['ver', 'name', 'time', 'data']

        for field in required_fields:
            if field not in raw_data:
                self.logger.log_error(
                    ValueError(f"Missing required field: {field}"),
                    {'context': 'validating_telemetry_data'}
                )
                return False

        # Check if data section has required operation field
        if 'Op' not in raw_data.get('data', {}):
            self.logger.log_error(
                ValueError("Missing required field 'Op' in data section"),
                {'context': 'validating_telemetry_data'}
            )
            return False

        return True

    def extract_key_metrics(self, event: TelemetryEvent) -> Dict[str, Any]:
        """
        Extract key metrics from a telemetry event

        Args:
            event: Parsed telemetry event

        Returns:
            Dictionary containing key metrics
        """
        return {
            'operation': event.operation,
            'pfn': event.pfn,
            'os_version': event.os_version,
            'app_version': event.app_version,
            'device_model': event.dev_model,
            'device_class': event.device_class,
            'user_timezone': event.tz,
            'event_sequence': event.seq,
            'epoch': event.epoch,
            'is_production_app': event.is_1p == 1,
            'data_category': event.data_category,
            'is_required_data': event.is_required
        }

    def batch_parse(self, telemetry_data_list: List[Dict[str, Any]]) -> List[TelemetryEvent]:
        """
        Parse multiple telemetry data entries

        Args:
            telemetry_data_list: List of raw telemetry data dictionaries

        Returns:
            List of parsed TelemetryEvent objects
        """
        parsed_events = []

        for i, raw_data in enumerate(telemetry_data_list):
            if self.validate_telemetry_data(raw_data):
                event = self.parse_telemetry_data(raw_data)
                if event:
                    parsed_events.append(event)

            # Log progress for large batches
            if (i + 1) % 100 == 0:
                self.logger.get_logger().info(f"Processed {i + 1} telemetry records")

        return parsed_events

# Global parser instance
telemetry_parser = TelemetryParser()
