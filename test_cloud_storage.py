"""
Test suite for Cloud Storage functionality
"""
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from src.cloud_storage import CloudStorageManager

class TestCloudStorageManager:
    """Test cases for CloudStorageManager"""

    def setup_method(self):
        """Set up test data and mocks"""
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

        # Mock credentials
        self.mock_credentials = {
            'aws_access_key': 'test_aws_key',
            'aws_secret_key': 'test_aws_secret',
            'aws_region': 'us-east-1',
            'gcs_project': 'test_project',
            'gcs_credentials_path': '/path/to/credentials.json',
            'azure_account_name': 'test_account',
            'azure_account_key': 'test_key'
        }

    @patch('boto3.client')
    def test_aws_s3_upload(self, mock_boto3_client):
        """Test AWS S3 upload functionality"""
        # Setup mock
        mock_s3_client = Mock()
        mock_boto3_client.return_value = mock_s3_client

        manager = CloudStorageManager(self.mock_credentials)

        # Test upload
        result = manager.upload_to_aws_s3(
            data=json.dumps(self.sample_data),
            bucket_name='test-bucket',
            filename='test_file.json'
        )

        assert result.startswith('SUCCESS')
        mock_s3_client.put_object.assert_called_once()

    @patch('google.cloud.storage.Client')
    def test_gcs_upload(self, mock_gcs_client):
        """Test Google Cloud Storage upload functionality"""
        # Setup mock
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_gcs_client.return_value = mock_client
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        manager = CloudStorageManager(self.mock_credentials)

        # Test upload
        result = manager.upload_to_gcs(
            data=json.dumps(self.sample_data),
            bucket_name='test-bucket',
            filename='test_file.json'
        )

        assert result.startswith('SUCCESS')
        mock_blob.upload_from_string.assert_called_once()

    @patch('azure.storage.blob.BlobServiceClient')
    def test_azure_upload(self, mock_blob_service):
        """Test Azure Blob Storage upload functionality"""
        # Setup mock
        mock_service_client = Mock()
        mock_container_client = Mock()
        mock_blob_client = Mock()
        mock_blob_service.return_value = mock_service_client
        mock_service_client.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client

        manager = CloudStorageManager(self.mock_credentials)

        # Test upload
        result = manager.upload_to_azure(
            data=json.dumps(self.sample_data),
            container_name='test-container',
            filename='test_file.json'
        )

        assert result.startswith('SUCCESS')
        mock_blob_client.upload_blob.assert_called_once()

    def test_export_telemetry_data(self):
        """Test telemetry data export to multiple providers"""
        manager = CloudStorageManager(self.mock_credentials)

        # Mock all upload methods
        with patch.object(manager, 'upload_to_aws_s3', return_value='SUCCESS: AWS upload complete'), \
            patch.object(manager, 'upload_to_gcs', return_value='SUCCESS: GCS upload complete'), \
            patch.object(manager, 'upload_to_azure', return_value='SUCCESS: Azure upload complete'):

            results = manager.export_telemetry_data(
                data=self.sample_data,
                filename_prefix='test_export',
                format_type='json',
                providers=['aws', 'gcs', 'azure']
            )

            assert 'aws' in results
            assert 'gcs' in results
            assert 'azure' in results
            assert results['aws'].startswith('SUCCESS')
            assert results['gcs'].startswith('SUCCESS')
            assert results['azure'].startswith('SUCCESS')

    def test_export_with_invalid_provider(self):
        """Test export with invalid provider"""
        manager = CloudStorageManager(self.mock_credentials)

        results = manager.export_telemetry_data(
            data=self.sample_data,
            filename_prefix='test_export',
            format_type='json',
            providers=['invalid_provider']
        )

        assert 'invalid_provider' in results
        assert results['invalid_provider'].startswith('ERROR')

    def test_export_empty_data(self):
        """Test export with empty data"""
        manager = CloudStorageManager(self.mock_credentials)

        results = manager.export_telemetry_data(
            data=[],
            filename_prefix='test_export',
            format_type='json',
            providers=['aws']
        )

        assert 'aws' in results
        assert results['aws'].startswith('ERROR')

    def test_get_supported_providers(self):
        """Test getting supported providers"""
        manager = CloudStorageManager(self.mock_credentials)
        providers = manager.get_supported_providers()

        assert 'aws' in providers
        assert 'gcs' in providers
        assert 'azure' in providers

    def test_provider_initialization(self):
        """Test provider initialization"""
        manager = CloudStorageManager(self.mock_credentials)

        assert hasattr(manager, 'providers')
        assert 'aws' in manager.providers
        assert 'gcs' in manager.providers
        assert 'azure' in manager.providers

    def test_credential_validation(self):
        """Test credential validation"""
        # Test with missing credentials
        incomplete_credentials = {
            'aws_access_key': 'test_key'
            # Missing other required credentials
        }

        with pytest.raises(Exception):
            CloudStorageManager(incomplete_credentials)

    def test_filename_generation(self):
        """Test filename generation for exports"""
        manager = CloudStorageManager(self.mock_credentials)

        # Test filename generation
        filename = manager._generate_filename('test_prefix', 'json')
        assert filename.startswith('test_prefix')
        assert filename.endswith('.json')
        assert 'telemetry_export' in filename

    def test_data_format_validation(self):
        """Test data format validation"""
        manager = CloudStorageManager(self.mock_credentials)

        # Test with valid format
        assert manager._validate_format('json') == True
        assert manager._validate_format('csv') == True

        # Test with invalid format
        assert manager._validate_format('invalid') == False

    def test_error_handling_aws(self):
        """Test error handling for AWS S3"""
        manager = CloudStorageManager(self.mock_credentials)

        with patch.object(manager, 'upload_to_aws_s3', side_effect=Exception('AWS Error')):
            result = manager.upload_to_aws_s3(
                data=json.dumps(self.sample_data),
                bucket_name='test-bucket',
                filename='test_file.json'
            )

            assert result.startswith('ERROR')

    def test_error_handling_gcs(self):
        """Test error handling for Google Cloud Storage"""
        manager = CloudStorageManager(self.mock_credentials)

        with patch.object(manager, 'upload_to_gcs', side_effect=Exception('GCS Error')):
            result = manager.upload_to_gcs(
                data=json.dumps(self.sample_data),
                bucket_name='test-bucket',
                filename='test_file.json'
            )

            assert result.startswith('ERROR')

    def test_error_handling_azure(self):
        """Test error handling for Azure Blob Storage"""
        manager = CloudStorageManager(self.mock_credentials)

        with patch.object(manager, 'upload_to_azure', side_effect=Exception('Azure Error')):
            result = manager.upload_to_azure(
                data=json.dumps(self.sample_data),
                container_name='test-container',
                filename='test_file.json'
            )

            assert result.startswith('ERROR')

    def test_large_data_handling(self):
        """Test handling of large datasets"""
        # Create large dataset
        large_data = []
        for i in range(1000):
            large_data.append({
                'id': i,
                'timestamp': f'2023-01-01T{i%24:02d}:00:00Z',
                'operation': f'operation_{i}',
                'data': {'value': i, 'metadata': {'index': i}}
            })

        manager = CloudStorageManager(self.mock_credentials)

        with patch.object(manager, 'upload_to_aws_s3', return_value='SUCCESS: Large data uploaded'):
            result = manager.upload_to_aws_s3(
                data=json.dumps(large_data),
                bucket_name='test-bucket',
                filename='large_file.json'
            )

            assert result.startswith('SUCCESS')

    def test_concurrent_uploads(self):
        """Test concurrent uploads to multiple providers"""
        manager = CloudStorageManager(self.mock_credentials)

        with patch.object(manager, 'upload_to_aws_s3', return_value='SUCCESS: AWS upload complete'), \
            patch.object(manager, 'upload_to_gcs', return_value='SUCCESS: GCS upload complete'), \
            patch.object(manager, 'upload_to_azure', return_value='SUCCESS: Azure upload complete'):

            results = manager.export_telemetry_data(
                data=self.sample_data,
                filename_prefix='concurrent_test',
                format_type='json',
                providers=['aws', 'gcs', 'azure']
            )

            # Verify all uploads completed
            successful_uploads = [r for r in results.values() if r.startswith('SUCCESS')]
            assert len(successful_uploads) == 3

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
