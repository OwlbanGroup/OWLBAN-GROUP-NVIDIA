from flask import jsonify, Response
from src.data_format_converter import DataFormatConverter
from datetime import datetime, timezone


def convert_data_format_logic(request_data):
    """Perform the conversion work for `/data/convert`.

    Returns either a Flask response tuple `(body, status, headers)` or a
    Flask `Response` object for binary payloads.
    """
    if not request_data or 'data' not in request_data:
        return jsonify({
            'error': 'No data provided for conversion',
            'status': 'error'
        }), 400

    data = request_data['data']
    from_format = request_data.get('from_format', 'json').lower()
    to_format = request_data.get('to_format', 'json').lower()
    options = request_data.get('options', {})

    if not isinstance(data, list):
        return jsonify({
            'error': 'Data must be a list of records',
            'status': 'error'
        }), 400

    if from_format not in DataFormatConverter.get_supported_import_formats():
        return jsonify({
            'error': f'Unsupported import format. Supported formats: {DataFormatConverter.get_supported_import_formats()}',
            'status': 'error'
        }), 400

    if to_format not in DataFormatConverter.get_supported_formats():
        return jsonify({
            'error': f'Unsupported export format. Supported formats: {DataFormatConverter.get_supported_formats()}',
            'status': 'error'
        }), 400

    # Convert from source format to internal representation
    if from_format == 'json':
        internal_data = data
    elif from_format == 'csv':
        internal_data = DataFormatConverter.convert_from_csv('\n'.join([','.join([str(v) for v in record.values()]) for record in data]))
    elif from_format == 'xml':
        xml_data = request_data.get('xml_data', '')
        internal_data = DataFormatConverter.convert_from_xml(xml_data)
    elif from_format == 'yaml':
        yaml_data = request_data.get('yaml_data', '')
        internal_data = DataFormatConverter.convert_from_yaml(yaml_data)
    else:
        return jsonify({
            'error': f'Unsupported conversion from {from_format}',
            'status': 'error'
        }), 400

    # Convert to target format
    if to_format == 'json':
        result = DataFormatConverter.convert_to_json(internal_data, pretty=options.get('pretty', True))
        content_type = 'application/json'
        return result, 200, {'Content-Type': content_type}
    elif to_format == 'csv':
        result = DataFormatConverter.convert_to_csv(internal_data)
        content_type = 'text/csv'
        return result, 200, {'Content-Type': content_type}
    elif to_format == 'xml':
        result = DataFormatConverter.convert_to_xml(internal_data)
        content_type = 'application/xml'
        return result, 200, {'Content-Type': content_type}
    elif to_format == 'yaml':
        result = DataFormatConverter.convert_to_yaml(internal_data)
        content_type = 'application/x-yaml'
        return result, 200, {'Content-Type': content_type}
    elif to_format == 'excel':
        result_bytes = DataFormatConverter.convert_to_excel(internal_data)
        return Response(result_bytes, status=200, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    elif to_format == 'parquet':
        result_bytes = DataFormatConverter.convert_to_parquet(internal_data)
        return Response(result_bytes, status=200, mimetype='application/octet-stream')
    else:
        return jsonify({
            'error': f'Unsupported conversion to {to_format}',
            'status': 'error'
        }), 400
