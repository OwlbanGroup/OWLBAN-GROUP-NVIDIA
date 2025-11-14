"""
Data Format Converter for Multiple Export Formats
"""
import json
import csv
import io
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any, List, Union
import logging
from pathlib import Path
import yaml
import pickle
import base64
import gzip
import zipfile
import tempfile
import os

logger = logging.getLogger(__name__)

class DataFormatConverter:
    """Converts telemetry data between various formats"""

    @staticmethod
    def convert_to_json(data: List[Dict[str, Any]], pretty: bool = True) -> str:
        """Convert data to JSON format"""
        try:
            if pretty:
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error converting to JSON: {str(e)}")
            raise

    @staticmethod
    def convert_to_csv(data: List[Dict[str, Any]]) -> str:
        """Convert data to CSV format"""
        try:
            if not data:
                return ""

            output = io.StringIO()
            fieldnames = data[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
            return output.getvalue()
        except Exception as e:
            logger.error(f"Error converting to CSV: {str(e)}")
            raise

    @staticmethod
    def convert_to_xml(data: List[Dict[str, Any]], root_name: str = "telemetry_data") -> str:
        """Convert data to XML format"""
        try:
            root = ET.Element(root_name)

            for record in data:
                record_element = ET.SubElement(root, "record")

                for key, value in record.items():
                    field_element = ET.SubElement(record_element, key)
                    # Convert value to string safely
                    if value is None:
                        field_element.text = ""
                    elif isinstance(value, (dict, list)):
                        field_element.text = json.dumps(value)
                    else:
                        field_element.text = str(value)

            # Convert to string with pretty formatting
            rough_string = ET.tostring(root, encoding='utf-8')
            reparsed = ET.fromstring(rough_string)

            # Pretty print
            from xml.dom import minidom
            pretty_xml = minidom.parseString(ET.tostring(reparsed, encoding='utf-8')).toprettyxml(indent="  ")

            return pretty_xml
        except Exception as e:
            logger.error(f"Error converting to XML: {str(e)}")
            raise

    @staticmethod
    def convert_to_yaml(data: List[Dict[str, Any]]) -> str:
        """Convert data to YAML format"""
        try:
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"Error converting to YAML: {str(e)}")
            raise

    @staticmethod
    def convert_to_pickle(data: List[Dict[str, Any]]) -> bytes:
        """Convert data to Pickle format"""
        try:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Error converting to Pickle: {str(e)}")
            raise

    @staticmethod
    def convert_to_messagepack(data: List[Dict[str, Any]]) -> bytes:
        """Convert data to MessagePack format"""
        try:
            import msgpack
            return msgpack.packb(data, use_bin_type=True)
        except ImportError:
            raise ImportError("msgpack is required for MessagePack conversion")
        except Exception as e:
            logger.error(f"Error converting to MessagePack: {str(e)}")
            raise

    @staticmethod
    def convert_to_parquet(data: List[Dict[str, Any]]) -> bytes:
        """Convert data to Parquet format"""
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Convert to PyArrow table
            table = pa.Table.from_pandas(df)

            # Write to buffer
            buffer = io.BytesIO()
            pq.write_table(table, buffer)

            return buffer.getvalue()
        except ImportError:
            raise ImportError("pandas and pyarrow are required for Parquet conversion")
        except Exception as e:
            logger.error(f"Error converting to Parquet: {str(e)}")
            raise

    @staticmethod
    def convert_to_avro(data: List[Dict[str, Any]]) -> bytes:
        """Convert data to Avro format"""
        try:
            import avro.schema
            import avro.io
            from avro.datafile import DataFileWriter
            from avro.io import DatumWriter

            # Define a simple schema (you might want to make this more sophisticated)
            schema = avro.schema.parse("""
            {
                "type": "record",
                "name": "TelemetryRecord",
                "fields": [
                    {"name": "timestamp", "type": "string"},
                    {"name": "operation", "type": "string"},
                    {"name": "status", "type": "string"},
                    {"name": "data", "type": "string"}
                ]
            }
            """)

            # Convert data to match schema
            avro_data = []
            for record in data:
                avro_record = {
                    "timestamp": record.get("timestamp", ""),
                    "operation": record.get("operation", ""),
                    "status": record.get("status", ""),
                    "data": json.dumps(record.get("data", {}))
                }
                avro_data.append(avro_record)

            # Write to buffer
            buffer = io.BytesIO()
            with DataFileWriter(buffer, DatumWriter(), schema) as writer:
                for record in avro_data:
                    writer.append(record)

            return buffer.getvalue()
        except ImportError:
            raise ImportError("avro-python3 is required for Avro conversion")
        except Exception as e:
            logger.error(f"Error converting to Avro: {str(e)}")
            raise

    @staticmethod
    def convert_to_excel(data: List[Dict[str, Any]]) -> bytes:
        """Convert data to Excel format"""
        try:
            import pandas as pd

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Write to buffer
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='TelemetryData', index=False)

            return buffer.getvalue()
        except ImportError:
            raise ImportError("pandas and openpyxl are required for Excel conversion")
        except Exception as e:
            logger.error(f"Error converting to Excel: {str(e)}")
            raise

    @staticmethod
    def convert_to_compressed_json(data: List[Dict[str, Any]], compression: str = "gzip") -> bytes:
        """Convert data to compressed JSON format"""
        try:
            json_data = DataFormatConverter.convert_to_json(data)

            if compression == "gzip":
                return gzip.compress(json_data.encode('utf-8'))
            elif compression == "zip":
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.writestr('data.json', json_data)
                return buffer.getvalue()
            else:
                raise ValueError(f"Unsupported compression: {compression}")
        except Exception as e:
            logger.error(f"Error converting to compressed JSON: {str(e)}")
            raise

    @staticmethod
    def convert_to_base64(data: List[Dict[str, Any]], format_type: str = "json") -> str:
        """Convert data to base64 encoded format"""
        try:
            if format_type == "json":
                json_data = DataFormatConverter.convert_to_json(data)
                return base64.b64encode(json_data.encode('utf-8')).decode('utf-8')
            elif format_type == "pickle":
                pickle_data = DataFormatConverter.convert_to_pickle(data)
                return base64.b64encode(pickle_data).decode('utf-8')
            else:
                raise ValueError(f"Unsupported format for base64: {format_type}")
        except Exception as e:
            logger.error(f"Error converting to base64: {str(e)}")
            raise

    @staticmethod
    def convert_from_json(json_data: str) -> List[Dict[str, Any]]:
        """Convert JSON data back to list of dictionaries"""
        try:
            return json.loads(json_data)
        except Exception as e:
            logger.error(f"Error converting from JSON: {str(e)}")
            raise

    @staticmethod
    def convert_from_csv(csv_data: str) -> List[Dict[str, Any]]:
        """Convert CSV data back to list of dictionaries"""
        try:
            reader = csv.DictReader(io.StringIO(csv_data))
            return list(reader)
        except Exception as e:
            logger.error(f"Error converting from CSV: {str(e)}")
            raise

    @staticmethod
    def convert_from_xml(xml_data: str) -> List[Dict[str, Any]]:
        """Convert XML data back to list of dictionaries"""
        try:
            root = ET.fromstring(xml_data)
            records = []

            for record_element in root.findall('record'):
                record = {}
                for field in record_element:
                    if field.text:
                        try:
                            # Try to parse as JSON if it looks like a complex object
                            record[field.tag] = json.loads(field.text)
                        except (json.JSONDecodeError, TypeError):
                            record[field.tag] = field.text
                    else:
                        record[field.tag] = ""
                records.append(record)

            return records
        except Exception as e:
            logger.error(f"Error converting from XML: {str(e)}")
            raise

    @staticmethod
    def convert_from_yaml(yaml_data: str) -> List[Dict[str, Any]]:
        """Convert YAML data back to list of dictionaries"""
        try:
            return yaml.safe_load(yaml_data)
        except Exception as e:
            logger.error(f"Error converting from YAML: {str(e)}")
            raise

    @staticmethod
    def convert_from_pickle(pickle_data: bytes) -> List[Dict[str, Any]]:
        """Convert Pickle data back to list of dictionaries"""
        try:
            return pickle.loads(pickle_data)
        except Exception as e:
            logger.error(f"Error converting from Pickle: {str(e)}")
            raise

    @staticmethod
    def convert_from_messagepack(msgpack_data: bytes) -> List[Dict[str, Any]]:
        """Convert MessagePack data back to list of dictionaries"""
        try:
            import msgpack
            return msgpack.unpackb(msgpack_data, raw=False)
        except ImportError:
            raise ImportError("msgpack is required for MessagePack conversion")
        except Exception as e:
            logger.error(f"Error converting from MessagePack: {str(e)}")
            raise

    @staticmethod
    def convert_from_parquet(parquet_data: bytes) -> List[Dict[str, Any]]:
        """Convert Parquet data back to list of dictionaries"""
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq

            buffer = io.BytesIO(parquet_data)
            table = pq.read_table(buffer)
            df = table.to_pandas()
            return df.to_dict('records')
        except ImportError:
            raise ImportError("pandas and pyarrow are required for Parquet conversion")
        except Exception as e:
            logger.error(f"Error converting from Parquet: {str(e)}")
            raise

    @staticmethod
    def convert_from_base64(base64_data: str, format_type: str = "json") -> List[Dict[str, Any]]:
        """Convert base64 data back to list of dictionaries"""
        try:
            decoded_data = base64.b64decode(base64_data)

            if format_type == "json":
                json_data = decoded_data.decode('utf-8')
                return DataFormatConverter.convert_from_json(json_data)
            elif format_type == "pickle":
                return DataFormatConverter.convert_from_pickle(decoded_data)
            else:
                raise ValueError(f"Unsupported format for base64: {format_type}")
        except Exception as e:
            logger.error(f"Error converting from base64: {str(e)}")
            raise

    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported export formats"""
        return [
            'json', 'csv', 'xml', 'yaml', 'pickle', 'messagepack',
            'parquet', 'avro', 'excel', 'compressed_json', 'base64'
        ]

    @staticmethod
    def get_supported_import_formats() -> List[str]:
        """Get list of supported import formats"""
        return [
            'json', 'csv', 'xml', 'yaml', 'pickle', 'messagepack',
            'parquet', 'base64'
        ]

    @staticmethod
    def validate_format(format_type: str, is_import: bool = False) -> bool:
        """Validate if a format is supported"""
        if is_import:
            return format_type.lower() in DataFormatConverter.get_supported_import_formats()
        else:
            return format_type.lower() in DataFormatConverter.get_supported_formats()

# Convenience functions for common conversions
def to_json(data: List[Dict[str, Any]], pretty: bool = True) -> str:
    """Convert data to JSON"""
    return DataFormatConverter.convert_to_json(data, pretty)

def to_csv(data: List[Dict[str, Any]]) -> str:
    """Convert data to CSV"""
    return DataFormatConverter.convert_to_csv(data)

def to_xml(data: List[Dict[str, Any]]) -> str:
    """Convert data to XML"""
    return DataFormatConverter.convert_to_xml(data)

def from_json(json_data: str) -> List[Dict[str, Any]]:
    """Convert JSON to data"""
    return DataFormatConverter.convert_from_json(json_data)

def from_csv(csv_data: str) -> List[Dict[str, Any]]:
    """Convert CSV to data"""
    return DataFormatConverter.convert_from_csv(csv_data)

if __name__ == "__main__":
    # Example usage
    sample_data = [
        {
            'timestamp': '2023-01-01T00:00:00Z',
            'operation': 'test_operation',
            'status': 'success',
            'data': {'key': 'value'}
        },
        {
            'timestamp': '2023-01-01T01:00:00Z',
            'operation': 'another_operation',
            'status': 'failed',
            'data': {'error': 'test error'}
        }
    ]

    # Convert to different formats
    json_data = DataFormatConverter.convert_to_json(sample_data)
    csv_data = DataFormatConverter.convert_to_csv(sample_data)
    xml_data = DataFormatConverter.convert_to_xml(sample_data)
    yaml_data = DataFormatConverter.convert_to_yaml(sample_data)

    print("JSON format:")
    print(json_data[:200] + "...")
    print("\nCSV format:")
    print(csv_data[:200] + "...")
    print("\nXML format:")
    print(xml_data[:200] + "...")
    print("\nYAML format:")
    print(yaml_data[:200] + "...")

    # Convert back
    parsed_json = DataFormatConverter.convert_from_json(json_data)
    parsed_csv = DataFormatConverter.convert_from_csv(csv_data)
    parsed_xml = DataFormatConverter.convert_from_xml(xml_data)
    parsed_yaml = DataFormatConverter.convert_from_yaml(yaml_data)

    print(f"\nParsed back {len(parsed_json)} records from JSON")
    print(f"Parsed back {len(parsed_csv)} records from CSV")
    print(f"Parsed back {len(parsed_xml)} records from XML")
    print(f"Parsed back {len(parsed_yaml)} records from YAML")
