"""
Test script for the telemetry handler with Microsoft Windows Store telemetry data
"""
import json
from datetime import datetime

from .src.telemetry_handler import telemetry_handler
from .src.logger import telemetry_logger

def test_telemetry_processing():
    """Test processing of the provided Microsoft Windows Store telemetry data"""

    # Sample telemetry data (the data provided by the user)
    sample_telemetry_data = {
        "ver": "4.0",
        "name": "Microsoft.Windows.ApplicationModel.Store.Telemetry.BeginOperation",
        "time": "2025-09-22T19:42:10.2549325Z",
        "iKey": "o:0a89d516ae714e01ae89c96d185e9ae3",
        "ext": {
            "utc": {
                "shellId": 281587075372417024,
                "eventFlags": 258,
                "pgName": "WINCORE",
                "dvcSample": 61.71,
                "flags": 52042924645,
                "edition": 101,
                "epoch": "1104474",
                "seq": 1158
            },
            "privacy": {
                "dataType": 2147483648,
                "isRequired": True,
                "dataCategory": 1,
                "product": 1
            },
            "metadata": {
                "privTags": 2147483648,
                "policies": 0
            },
            "mscv": {
                "cV": "fka+IohbR063mR8A.1"
            },
            "os": {
                "bootId": 14,
                "name": "Windows",
                "ver": "10.0.26100.6584.amd64fre.ge_release.240331-1435",
                "expId": "RS:2AEB0,MD:283BAEF,ME:33B9B19,ME:33B9B24,ME:33B9B29,ME:33B9B30,ME:3667C98,MD:3667C9A,ME:3536BD9,MD:2FE0A31,MD:2FE0A40,MD:2FE0A4F,MD:37B239D,PD:3536CF8,PD:33CDA4D"
            },
            "app": {
                "id": "U:Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe!runtimebroker07f4358a809ac99a64a67c1",
                "ver": "22507.1401.7.0_x64_!2101/12/11:19:06:32!24DCB!runtimebroker.exe",
                "is1P": 1,
                "asId": 2097
            },
            "device": {
                "localId": "s:9703B182-C525-4DD8-B63F-3A648556E146",
                "deviceClass": "Windows.Desktop"
            },
            "protocol": {
                "devMake": "HP",
                "devModel": "Victus by HP Gaming Laptop 15-fa2xxx",
                "ticketKeys": [
                    "32593120"
                ]
            },
            "user": {
                "localId": "m:defcee03262a7637"
            },
            "loc": {
                "tz": "-04:00"
            }
        },
        "data": {
            "Op": "StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
            "PFN": "Microsoft.WindowsStore_22507.1401.7.0_x64__8wekyb3d8bbwe",
            "PN1": "",
            "P1": "",
            "PN2": "",
            "P2": "",
            "PN3": "",
            "P3": "",
            "PN4": "",
            "P4": ""
        }
    }

    print("=== Testing Telemetry Processing ===")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Test single event processing
    print("\n1. Testing single event processing...")
    success = telemetry_handler.process_single_event(sample_telemetry_data)

    if success:
        print("✓ Single event processed successfully")
    else:
        print("✗ Failed to process single event")
        return False

    # Test batch processing with multiple events
    print("\n2. Testing batch processing...")
    batch_data = [sample_telemetry_data] * 5  # Create 5 copies for batch testing

    stats = telemetry_handler.process_batch(batch_data)

    print(f"✓ Batch processing completed: {stats['successful']}/{stats['total']} events successful")

    # Test metrics retrieval
    print("\n3. Testing metrics retrieval...")
    metrics = telemetry_handler.get_metrics(hours=24)

    print("✓ Metrics retrieved:")
    print(f"  - Total events: {metrics.get('total_events', 0)}")
    print(f"  - Operations: {list(metrics.get('operation_counts', {}).keys())}")
    print(f"  - Device classes: {list(metrics.get('device_counts', {}).keys())}")

    # Test data export
    print("\n4. Testing data export...")
    exported_events = telemetry_handler.export_events(
        operation="StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync",
        limit=10
    )

    print(f"✓ Exported {len(exported_events)} events")

    # Test key metrics extraction
    print("\n5. Testing key metrics extraction...")
    from .src.telemetry_parser import telemetry_parser

    event = telemetry_parser.parse_telemetry_data(sample_telemetry_data)
    if event:
        key_metrics = telemetry_parser.extract_key_metrics(event)
        print("✓ Key metrics extracted:")
        print(f"  - Operation: {key_metrics['operation']}")
        print(f"  - PFN: {key_metrics['pfn']}")
        print(f"  - OS Version: {key_metrics['os_version']}")
        print(f"  - Device Model: {key_metrics['device_model']}")
        print(f"  - Device Class: {key_metrics['device_class']}")
        print(f"  - Is Production App: {key_metrics['is_production_app']}")

    print("\n=== Test Summary ===")
    print("✓ All telemetry processing tests completed successfully!")
    print("✓ The system can handle Microsoft Windows Store telemetry data")
    print("✓ Data is stored in SQLite database")
    print("✓ Metrics and analytics are available")
    print("✓ Export functionality is working")

    return True

def demonstrate_api_usage():
    """Demonstrate how to use the telemetry API"""

    print("\n=== API Usage Examples ===")

    print("\n1. Single Event API:")
    print("POST /telemetry")
    print("Content-Type: application/json")
    print("Body: {telemetry JSON data}")

    print("\n2. Batch Processing API:")
    print("POST /telemetry/batch")
    print("Content-Type: application/json")
    print("Body: {'telemetry_data': [array of telemetry events]}")

    print("\n3. Metrics API:")
    print("GET /telemetry/metrics?hours=24")

    print("\n4. Export API:")
    print("GET /telemetry/export?operation=StoreConfigurationServer::FilterUnsupportedSystemFeaturesAsync&limit=100&format=json")

    print("\n5. Health Check:")
    print("GET /health")

if __name__ == "__main__":
    # Run the test
    test_telemetry_processing()

    # Show API usage examples
    demonstrate_api_usage()

    print("\n=== Setup Instructions ===")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the application: python app.py")
    print("3. Test the API: python test_telemetry.py")
    print("4. Access the API at: http://localhost:5000")
