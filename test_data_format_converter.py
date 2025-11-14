"""
Test suite for Data Format Converter functionality
"""
import pytest
import json
import tempfile
import os
from src.data_format_converter import DataFormatConverter

class TestDataFormatConverter:
    """Test cases for DataFormatConverter"""

    def setup_method(self):
        """Set up test data"""
        self.sample_data = [
            {
                'timestamp': '2023-01-01T00:00:00Z',
                'operation': 'test_operation',
                'status': 'success',
                'data': {'key': 'value', 'number': 42}
            },
            {
                'timestamp': '2023-01-01T01:00:00Z',
                'operation': 'another_operation',
                'status': 'failed',
                'data': {'error': 'test error', 'code': 500}
            }
        ]

    def test_convert_to_json(self):
        """Test JSON conversion"""
        result = DataFormatConverter.convert_to_json(self.sample_data)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]['operation'] == 'test_operation'

    def test_convert_to_json_pretty(self):
        """Test pretty JSON conversion"""
        result = DataFormatConverter.convert_to_json(self.sample_data, pretty=True)
        assert '\n' in result
        assert '  ' in result  # Indentation

    def test_convert_to_csv(self):
        """Test CSV conversion"""
        result = DataFormatConverter.convert_to_csv(self.sample_data)
        assert 'timestamp' in result
        assert 'operation' in result
        assert 'test_operation' in result

    def test_convert_to_xml(self):
        """Test XML conversion"""
        result = DataFormatConverter.convert_to_xml(self.sample_data)
        assert '<telemetry_data>' in result
        assert '<record>' in result
        assert 'test_operation' in result

    def test_convert_to_yaml(self):
        """Test YAML conversion"""
        result = DataFormatConverter.convert_to_yaml(self.sample_data)
        assert 'timestamp:' in result
        assert 'operation:' in result
        assert 'test_operation' in result

    def test_convert_from_json(self):
        """Test JSON parsing"""
        json_data = DataFormatConverter.convert_to_json(self.sample_data)
        parsed = DataFormatConverter.convert_from_json(json_data)
        assert len(parsed) == 2
        assert parsed[0]['operation'] == 'test_operation'

    def test_convert_from_csv(self):
        """Test CSV parsing"""
        csv_data = DataFormatConverter.convert_to_csv(self.sample_data)
        parsed = DataFormatConverter.convert_from_csv(csv_data)
        assert len(parsed) == 2
        assert parsed[0]['operation'] == 'test_operation'

    def test_convert_from_xml(self):
        """Test XML parsing"""
        xml_data = DataFormatConverter.convert_to_xml(self.sample_data)
        parsed = DataFormatConverter.convert_from_xml(xml_data)
        assert len(parsed) == 2
        assert parsed[0]['operation'] == 'test_operation'

    def test_convert_from_yaml(self):
        """Test YAML parsing"""
        yaml_data = DataFormatConverter.convert_to_yaml(self.sample_data)
        parsed = DataFormatConverter.convert_from_yaml(yaml_data)
        assert len(parsed) == 2
        assert parsed[0]['operation'] == 'test_operation'

    def test_get_supported_formats(self):
        """Test getting supported formats"""
        export_formats = DataFormatConverter.get_supported_formats()
        import_formats = DataFormatConverter.get_supported_import_formats()

        assert 'json' in export_formats
        assert 'csv' in export_formats
        assert 'xml' in export_formats
        assert 'yaml' in export_formats

        assert 'json' in import_formats
        assert 'csv' in import_formats
        assert 'xml' in import_formats
        assert 'yaml' in import_formats

    def test_validate_format(self):
        """Test format validation"""
        assert DataFormatConverter.validate_format('json') == True
        assert DataFormatConverter.validate_format('csv') == True
        assert DataFormatConverter.validate_format('xml') == True
        assert DataFormatConverter.validate_format('invalid') == False

    def test_empty_data_conversion(self):
        """Test conversion with empty data"""
        empty_data = []

        json_result = DataFormatConverter.convert_to_json(empty_data)
        assert json_result == '[]'

        csv_result = DataFormatConverter.convert_to_csv(empty_data)
        assert csv_result == ''

    def test_complex_data_conversion(self):
        """Test conversion with complex nested data"""
        complex_data = [
            {
                'metadata': {
                    'version': '1.0',
                    'tags': ['tag1', 'tag2'],
                    'config': {'nested': {'value': 123}}
                },
                'data': [1, 2, 3, {'nested': 'value'}]
            }
        ]

        # Test JSON conversion
        json_result = DataFormatConverter.convert_to_json(complex_data)
        parsed = json.loads(json_result)
        assert parsed[0]['metadata']['version'] == '1.0'

        # Test XML conversion
        xml_result = DataFormatConverter.convert_to_xml(complex_data)
        assert '<metadata>' in xml_result
        assert '<config>' in xml_result

    def test_error_handling(self):
        """Test error handling in conversions"""
        # Test with None data
        with pytest.raises(Exception):
            DataFormatConverter.convert_to_json(None)

        # Test with invalid data structure
        with pytest.raises(Exception):
            DataFormatConverter.convert_to_csv([{'key': 'value'}, 'invalid'])

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion (format -> parse -> format)"""
        original_data = self.sample_data.copy()

        # JSON roundtrip
        json_data = DataFormatConverter.convert_to_json(original_data)
        parsed_json = DataFormatConverter.convert_from_json(json_data)
        back_to_json = DataFormatConverter.convert_to_json(parsed_json)
        assert json.loads(json_data) == json.loads(back_to_json)

        # CSV roundtrip
        csv_data = DataFormatConverter.convert_to_csv(original_data)
        parsed_csv = DataFormatConverter.convert_from_csv(csv_data)
        back_to_csv = DataFormatConverter.convert_to_csv(parsed_csv)
        assert csv_data == back_to_csv

    def test_format_consistency(self):
        """Test that different formats contain the same data"""
        json_data = DataFormatConverter.convert_to_json(self.sample_data)
        csv_data = DataFormatConverter.convert_to_csv(self.sample_data)
        xml_data = DataFormatConverter.convert_to_xml(self.sample_data)
        yaml_data = DataFormatConverter.convert_to_yaml(self.sample_data)

        # Parse all formats back
        parsed_json = DataFormatConverter.convert_from_json(json_data)
        parsed_csv = DataFormatConverter.convert_from_csv(csv_data)
        parsed_xml = DataFormatConverter.convert_from_xml(xml_data)
        parsed_yaml = DataFormatConverter.convert_from_yaml(yaml_data)

        # All should have the same core data
        assert len(parsed_json) == len(parsed_csv) == len(parsed_xml) == len(parsed_yaml)
        assert parsed_json[0]['operation'] == parsed_csv[0]['operation'] == parsed_xml[0]['operation'] == parsed_yaml[0]['operation']

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
