#!/usr/bin/env python3
"""
Comprehensive tests for the new /telemetry/export and /ml/train endpoints
"""
import pytest
import json
import io
from unittest.mock import patch, MagicMock
from app_final import app


@pytest.fixture
def client():
    """Test client fixture"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestTelemetryExport:
    """Test cases for /telemetry/export endpoint"""

    @patch('app_final.telemetry_handler.export_events')
    def test_export_telemetry_json_default(self, mock_export_events, client):
        """Test exporting telemetry data in JSON format with default parameters"""
        mock_events = [
            {'operation': 'test_op', 'timestamp': '2023-01-01T00:00:00Z', 'data': {'key': 'value'}},
            {'operation': 'test_op2', 'timestamp': '2023-01-01T00:01:00Z', 'data': {'key2': 'value2'}}
        ]
        mock_export_events.return_value = mock_events

        response = client.get('/telemetry/export')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['count'] == 2
        assert len(data['events']) == 2
        assert 'timestamp' in data
        mock_export_events.assert_called_once_with(operation=None, limit=1000)

    @patch('app_final.telemetry_handler.export_events')
    def test_export_telemetry_with_parameters(self, mock_export_events, client):
        """Test exporting telemetry data with custom parameters"""
        mock_events = [{'operation': 'specific_op', 'timestamp': '2023-01-01T00:00:00Z'}]
        mock_export_events.return_value = mock_events

        response = client.get('/telemetry/export?operation=specific_op&limit=500&format=json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['count'] == 1
        mock_export_events.assert_called_once_with(operation='specific_op', limit=500)

    @patch('app_final.telemetry_handler.export_events')
    def test_export_telemetry_csv_format(self, mock_export_events, client):
        """Test exporting telemetry data in CSV format"""
        mock_events = [
            {'operation': 'test_op', 'timestamp': '2023-01-01T00:00:00Z', 'value': 123},
            {'operation': 'test_op2', 'timestamp': '2023-01-01T00:01:00Z', 'value': 456}
        ]
        mock_export_events.return_value = mock_events

        response = client.get('/telemetry/export?format=csv')

        assert response.status_code == 200
        assert response.content_type == 'text/csv'
        assert 'attachment; filename=telemetry_export.csv' in response.headers.get('Content-Disposition', '')

        # Parse CSV content
        csv_content = response.data.decode('utf-8')
        lines = csv_content.strip().split('\n')
        assert len(lines) == 3  # Header + 2 data rows
        assert 'operation,timestamp,value' in lines[0]

    @patch('app_final.telemetry_handler.export_events')
    def test_export_telemetry_empty_results_csv(self, mock_export_events, client):
        """Test exporting empty telemetry data in CSV format"""
        mock_export_events.return_value = []

        response = client.get('/telemetry/export?format=csv')

        assert response.status_code == 200
        assert response.content_type == 'text/csv'

    def test_export_telemetry_invalid_limit_high(self, client):
        """Test exporting with limit too high"""
        response = client.get('/telemetry/export?limit=15000')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Limit must be between 1 and 10000' in data['error']

    def test_export_telemetry_invalid_limit_low(self, client):
        """Test exporting with limit too low"""
        response = client.get('/telemetry/export?limit=0')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Limit must be between 1 and 10000' in data['error']

    def test_export_telemetry_invalid_format(self, client):
        """Test exporting with invalid format"""
        response = client.get('/telemetry/export?format=xml')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Format must be json or csv' in data['error']

    @patch('app_final.telemetry_handler.export_events')
    def test_export_telemetry_server_error(self, mock_export_events, client):
        """Test exporting with server error"""
        mock_export_events.side_effect = Exception("Database connection failed")

        response = client.get('/telemetry/export')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Internal server error' in data['error']


class TestMLTrain:
    """Test cases for /ml/train endpoint"""

    @patch('app_final.anomaly_detector.train')
    @patch('app_final.token_manager.validate_token')
    def test_train_ml_model_success(self, mock_validate_token, mock_train, client):
        """Test successful ML model training"""
        mock_validate_token.return_value = True

        training_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        payload = {
            'training_data': training_data,
            'contamination': 0.1
        }

        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json',
                                headers={'Authorization': 'Bearer test_token'})

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'ML model trained successfully' in data['message']
        assert data['samples_used'] == 3
        assert data['contamination'] == 0.1
        assert 'timestamp' in data
        mock_train.assert_called_once()

    @patch('app_final.token_manager.validate_token')
    def test_train_ml_model_unauthorized(self, mock_validate_token, client):
        """Test ML training without authentication"""
        mock_validate_token.return_value = False

        payload = {'training_data': [[1, 2, 3]]}

        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json',
                                headers={'Authorization': 'Bearer invalid_token'})

        assert response.status_code == 401
        data = json.loads(response.data)
        assert 'Missing or invalid authorization header' in data['error']

    def test_train_ml_model_no_auth_header(self, client):
        """Test ML training without authorization header"""
        payload = {'training_data': [[1, 2, 3]]}

        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json')

        assert response.status_code == 401
        data = json.loads(response.data)
        assert 'Missing or invalid authorization header' in data['error']

    @patch('app_final.token_manager.validate_token')
    def test_train_ml_model_missing_data(self, mock_validate_token, client):
        """Test ML training with missing training data"""
        mock_validate_token.return_value = True

        payload = {}  # Missing training_data

        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json',
                                headers={'Authorization': 'Bearer test_token'})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'No training data provided' in data['error']

    @patch('app_final.token_manager.validate_token')
    def test_train_ml_model_invalid_data_type(self, mock_validate_token, client):
        """Test ML training with invalid training data type"""
        mock_validate_token.return_value = True

        payload = {'training_data': 'not_a_list'}  # Should be a list

        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json',
                                headers={'Authorization': 'Bearer test_token'})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Training data must be a list' in data['error']

    @patch('app_final.token_manager.validate_token')
    def test_train_ml_model_insufficient_data(self, mock_validate_token, client):
        """Test ML training with insufficient training data"""
        mock_validate_token.return_value = True

        payload = {'training_data': [[1, 2, 3]]}  # Only 1 sample, need at least 10

        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json',
                                headers={'Authorization': 'Bearer test_token'})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Training data must be a list with at least 10 samples' in data['error']

    @patch('app_final.token_manager.validate_token')
    def test_train_ml_model_invalid_contamination_high(self, mock_validate_token, client):
        """Test ML training with contamination too high"""
        mock_validate_token.return_value = True

        training_data = [[i, i+1, i+2] for i in range(10)]
        payload = {
            'training_data': training_data,
            'contamination': 0.8  # Too high, should be < 0.5
        }

        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json',
                                headers={'Authorization': 'Bearer test_token'})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Contamination must be between 0 and 0.5' in data['error']

    @patch('app_final.token_manager.validate_token')
    def test_train_ml_model_invalid_contamination_low(self, mock_validate_token, client):
        """Test ML training with contamination too low"""
        mock_validate_token.return_value = True

        training_data = [[i, i+1, i+2] for i in range(10)]
        payload = {
            'training_data': training_data,
            'contamination': -0.1  # Too low, should be > 0
        }

        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json',
                                headers={'Authorization': 'Bearer test_token'})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Contamination must be between 0 and 0.5' in data['error']

    @patch('app_final.anomaly_detector.train')
    @patch('app_final.token_manager.validate_token')
    def test_train_ml_model_default_contamination(self, mock_validate_token, mock_train, client):
        """Test ML training with default contamination value"""
        mock_validate_token.return_value = True

        training_data = [[i, i+1, i+2] for i in range(10)]
        payload = {'training_data': training_data}  # No contamination specified

        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json',
                                headers={'Authorization': 'Bearer test_token'})

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['contamination'] == 0.1  # Default value
        mock_train.assert_called_once()

    @patch('app_final.anomaly_detector.train')
    @patch('app_final.token_manager.validate_token')
    def test_train_ml_model_server_error(self, mock_validate_token, mock_train, client):
        """Test ML training with server error"""
        mock_validate_token.return_value = True
        mock_train.side_effect = ValueError("Invalid training data format")

        training_data = [[i, i+1, i+2] for i in range(10)]
        payload = {'training_data': training_data}

        response = client.post('/ml/train',
                                data=json.dumps(payload),
                                content_type='application/json',
                                headers={'Authorization': 'Bearer test_token'})

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid training data format' in str(data['error'])

    def test_train_ml_model_invalid_json(self, client):
        """Test ML training with invalid JSON"""
        response = client.post('/ml/train',
                                data='invalid json',
                                content_type='application/json')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Invalid JSON format' in data['error']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
