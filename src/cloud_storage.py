"""
Cloud Storage Integration for Data Export
"""
import os
import json
import csv
import io
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from .circuit_breaker import CircuitBreaker

# Cloud storage imports
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class CloudStorageBase(ABC):
    """Base class for cloud storage providers"""

    @abstractmethod
    def upload_data(self, data: Any, filename: str, metadata: Optional[Dict] = None) -> str:
        """Upload data to cloud storage"""
        pass

    @abstractmethod
    def download_data(self, filename: str) -> Any:
        """Download data from cloud storage"""
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in storage"""
        pass

    @abstractmethod
    def delete_file(self, filename: str) -> bool:
        """Delete file from storage"""
        pass

class AWSStorage(CloudStorageBase):
    """AWS S3 storage implementation"""

    def __init__(self, bucket_name: str, aws_access_key: str = None, aws_secret_key: str = None, region: str = "us-east-1"):
        if not AWS_AVAILABLE:
            raise ImportError("boto3 is required for AWS storage")

        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region
        )

    def upload_data(self, data: Any, filename: str, metadata: Optional[Dict] = None) -> str:
        """Upload data to S3"""
        try:
            if isinstance(data, str):
                # Assume it's a string/JSON
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=data,
                    ContentType='application/json',
                    Metadata=metadata or {}
                )
            elif isinstance(data, bytes):
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=data,
                    Metadata=metadata or {}
                )
            else:
                # Convert to JSON
                json_data = json.dumps(data, indent=2)
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=filename,
                    Body=json_data,
                    ContentType='application/json',
                    Metadata=metadata or {}
                )

            logger.info(f"Successfully uploaded {filename} to S3")
            return f"s3://{self.bucket_name}/{filename}"

        except ClientError as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise

    def download_data(self, filename: str) -> str:
        """Download data from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
            return response['Body'].read().decode('utf-8')

        except ClientError as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            raise

    def list_files(self, prefix: str = "") -> List[str]:
        """List files in S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []

        except ClientError as e:
            logger.error(f"Error listing S3 files: {str(e)}")
            raise

    def delete_file(self, filename: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
            logger.info(f"Successfully deleted {filename} from S3")
            return True

        except ClientError as e:
            logger.error(f"Error deleting from S3: {str(e)}")
            return False

class GCSStorage(CloudStorageBase):
    """Google Cloud Storage implementation"""

    def __init__(self, bucket_name: str, credentials_path: str = None):
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage is required for GCS")

        self.bucket_name = bucket_name
        self.client = storage.Client.from_service_account_json(credentials_path) if credentials_path else storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload_data(self, data: Any, filename: str, metadata: Optional[Dict] = None) -> str:
        """Upload data to GCS"""
        try:
            blob = self.bucket.blob(filename)

            if isinstance(data, str):
                blob.upload_from_string(data, content_type='application/json')
            elif isinstance(data, bytes):
                blob.upload_from_string(data)
            else:
                json_data = json.dumps(data, indent=2)
                blob.upload_from_string(json_data, content_type='application/json')

            if metadata:
                blob.metadata = metadata
                blob.patch()

            logger.info(f"Successfully uploaded {filename} to GCS")
            return f"gs://{self.bucket_name}/{filename}"

        except Exception as e:
            logger.error(f"Error uploading to GCS: {str(e)}")
            raise

    def download_data(self, filename: str) -> str:
        """Download data from GCS"""
        try:
            blob = self.bucket.blob(filename)
            return blob.download_as_text()

        except Exception as e:
            logger.error(f"Error downloading from GCS: {str(e)}")
            raise

    def list_files(self, prefix: str = "") -> List[str]:
        """List files in GCS bucket"""
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]

        except Exception as e:
            logger.error(f"Error listing GCS files: {str(e)}")
            raise

    def delete_file(self, filename: str) -> bool:
        """Delete file from GCS"""
        try:
            blob = self.bucket.blob(filename)
            blob.delete()
            logger.info(f"Successfully deleted {filename} from GCS")
            return True

        except Exception as e:
            logger.error(f"Error deleting from GCS: {str(e)}")
            return False

class AzureStorage(CloudStorageBase):
    """Azure Blob Storage implementation"""

    def __init__(self, container_name: str, connection_string: str = None):
        if not AZURE_AVAILABLE:
            raise ImportError("azure-storage-blob is required for Azure storage")

        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string or os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def upload_data(self, data: Any, filename: str, metadata: Optional[Dict] = None) -> str:
        """Upload data to Azure Blob Storage"""
        try:
            blob_client = self.container_client.get_blob_client(filename)

            if isinstance(data, str):
                blob_client.upload_blob(data, overwrite=True, content_type='application/json')
            elif isinstance(data, bytes):
                blob_client.upload_blob(data, overwrite=True)
            else:
                json_data = json.dumps(data, indent=2)
                blob_client.upload_blob(json_data, overwrite=True, content_type='application/json')

            if metadata:
                blob_client.set_blob_metadata(metadata)

            logger.info(f"Successfully uploaded {filename} to Azure Blob Storage")
            return f"azure://{self.container_name}/{filename}"

        except Exception as e:
            logger.error(f"Error uploading to Azure: {str(e)}")
            raise

    def download_data(self, filename: str) -> str:
        """Download data from Azure Blob Storage"""
        try:
            blob_client = self.container_client.get_blob_client(filename)
            return blob_client.download_blob().readall().decode('utf-8')

        except Exception as e:
            logger.error(f"Error downloading from Azure: {str(e)}")
            raise

    def list_files(self, prefix: str = "") -> List[str]:
        """List files in Azure container"""
        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blobs]

        except Exception as e:
            logger.error(f"Error listing Azure files: {str(e)}")
            raise

    def delete_file(self, filename: str) -> bool:
        """Delete file from Azure Blob Storage"""
        try:
            blob_client = self.container_client.get_blob_client(filename)
            blob_client.delete_blob()
            logger.info(f"Successfully deleted {filename} from Azure Blob Storage")
            return True

        except Exception as e:
            logger.error(f"Error deleting from Azure: {str(e)}")
            return False

class MinIOStorage(CloudStorageBase):
    """MinIO (S3-compatible) storage implementation"""

    def __init__(self, endpoint: str, bucket_name: str, access_key: str = None, secret_key: str = None):
        if not MINIO_AVAILABLE:
            raise ImportError("minio is required for MinIO storage")

        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.client = Minio(
            endpoint,
            access_key=access_key or os.getenv('MINIO_ACCESS_KEY'),
            secret_key=secret_key or os.getenv('MINIO_SECRET_KEY'),
            secure=False  # Set to True for HTTPS
        )

        # Create bucket if it doesn't exist
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

    def upload_data(self, data: Any, filename: str, metadata: Optional[Dict] = None) -> str:
        """Upload data to MinIO"""
        try:
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                json_data = json.dumps(data, indent=2)
                data_bytes = json_data.encode('utf-8')

            self.client.put_object(
                self.bucket_name,
                filename,
                io.BytesIO(data_bytes),
                len(data_bytes),
                content_type='application/json',
                metadata=metadata
            )

            logger.info(f"Successfully uploaded {filename} to MinIO")
            return f"minio://{self.bucket_name}/{filename}"

        except Exception as e:
            logger.error(f"Error uploading to MinIO: {str(e)}")
            raise

    def download_data(self, filename: str) -> str:
        """Download data from MinIO"""
        try:
            response = self.client.get_object(self.bucket_name, filename)
            return response.read().decode('utf-8')

        except Exception as e:
            logger.error(f"Error downloading from MinIO: {str(e)}")
            raise

    def list_files(self, prefix: str = "") -> List[str]:
        """List files in MinIO bucket"""
        try:
            objects = self.client.list_objects(self.bucket_name, prefix=prefix)
            return [obj.object_name for obj in objects]

        except Exception as e:
            logger.error(f"Error listing MinIO files: {str(e)}")
            raise

    def delete_file(self, filename: str) -> bool:
        """Delete file from MinIO"""
        try:
            self.client.remove_object(self.bucket_name, filename)
            logger.info(f"Successfully deleted {filename} from MinIO")
            return True

        except Exception as e:
            logger.error(f"Error deleting from MinIO: {str(e)}")
            return False

class CloudStorageManager:
    """Manager for multiple cloud storage providers"""

    def __init__(self):
        self.providers: Dict[str, CloudStorageBase] = {}
        # Circuit breaker for cloud storage operations
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

    def add_provider(self, name: str, provider: CloudStorageBase):
        """Add a storage provider"""
        self.providers[name] = provider
        logger.info(f"Added storage provider: {name}")

    def export_telemetry_data(self, data: List[Dict], filename_prefix: str = "telemetry_export",
                            format_type: str = "json", providers: List[str] = None) -> Dict[str, str]:
        """
        Export telemetry data to multiple cloud storage providers

        Args:
            data: List of telemetry data dictionaries
            filename_prefix: Prefix for the filename
            format_type: Export format (json, csv)
            providers: List of provider names to export to (default: all)

        Returns:
            Dictionary mapping provider names to URLs
        """
        if providers is None:
            providers = list(self.providers.keys())

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results = {}

        for provider_name in providers:
            if provider_name not in self.providers:
                logger.warning(f"Provider {provider_name} not found, skipping")
                continue

            try:
                provider = self.providers[provider_name]

                if format_type.lower() == "csv":
                    filename = f"{filename_prefix}_{timestamp}.csv"
                    # Convert to CSV
                    if data:
                        output = io.StringIO()
                        fieldnames = data[0].keys()
                        writer = csv.DictWriter(output, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(data)
                        export_data = output.getvalue()
                    else:
                        export_data = ""
                else:
                    filename = f"{filename_prefix}_{timestamp}.json"
                    export_data = json.dumps(data, indent=2)

                # Add metadata
                metadata = {
                    'export_timestamp': datetime.now(timezone.utc).isoformat(),
                    'record_count': str(len(data)),
                    'format': format_type,
                    'source': 'jpmorgan_telemetry_api'
                }

                # Upload to provider
                url = provider.upload_data(export_data, filename, metadata)
                results[provider_name] = url

                logger.info(f"Successfully exported data to {provider_name}: {url}")

            except Exception as e:
                logger.error(f"Error exporting to {provider_name}: {str(e)}")
                results[provider_name] = f"ERROR: {str(e)}"

        return results

    def get_provider(self, name: str) -> Optional[CloudStorageBase]:
        """Get a storage provider by name"""
        return self.providers.get(name)

# Global cloud storage manager
cloud_storage_manager = CloudStorageManager()

def setup_cloud_storage(config: Dict[str, Any]):
    """Setup cloud storage providers from configuration"""
    try:
        # AWS S3
        if config.get('AWS_BUCKET_NAME'):
            try:
                aws_storage = AWSStorage(
                    bucket_name=config['AWS_BUCKET_NAME'],
                    aws_access_key=config.get('AWS_ACCESS_KEY'),
                    aws_secret_key=config.get('AWS_SECRET_KEY'),
                    region=config.get('AWS_REGION', 'us-east-1')
                )
                cloud_storage_manager.add_provider('aws', aws_storage)
                logger.info("AWS S3 storage configured")
            except Exception as e:
                logger.warning(f"Failed to configure AWS storage: {str(e)}")

        # Google Cloud Storage
        if config.get('GCS_BUCKET_NAME'):
            try:
                gcs_storage = GCSStorage(
                    bucket_name=config['GCS_BUCKET_NAME'],
                    credentials_path=config.get('GCS_CREDENTIALS_PATH')
                )
                cloud_storage_manager.add_provider('gcs', gcs_storage)
                logger.info("Google Cloud Storage configured")
            except Exception as e:
                logger.warning(f"Failed to configure GCS storage: {str(e)}")

        # Azure Blob Storage
        if config.get('AZURE_CONTAINER_NAME'):
            try:
                azure_storage = AzureStorage(
                    container_name=config['AZURE_CONTAINER_NAME'],
                    connection_string=config.get('AZURE_CONNECTION_STRING')
                )
                cloud_storage_manager.add_provider('azure', azure_storage)
                logger.info("Azure Blob Storage configured")
            except Exception as e:
                logger.warning(f"Failed to configure Azure storage: {str(e)}")

        # MinIO
        if config.get('MINIO_ENDPOINT'):
            try:
                minio_storage = MinIOStorage(
                    endpoint=config['MINIO_ENDPOINT'],
                    bucket_name=config['MINIO_BUCKET_NAME'],
                    access_key=config.get('MINIO_ACCESS_KEY'),
                    secret_key=config.get('MINIO_SECRET_KEY')
                )
                cloud_storage_manager.add_provider('minio', minio_storage)
                logger.info("MinIO storage configured")
            except Exception as e:
                logger.warning(f"Failed to configure MinIO storage: {str(e)}")

    except Exception as e:
        logger.error(f"Error setting up cloud storage: {str(e)}")

if __name__ == "__main__":
    # Example usage
    from config import config

    setup_cloud_storage(config.get_all_settings())

    # Example data
    sample_data = [
        {
            'timestamp': '2023-01-01T00:00:00Z',
            'operation': 'test_operation',
            'status': 'success'
        }
    ]

    # Export to all configured providers
    results = cloud_storage_manager.export_telemetry_data(
        data=sample_data,
        filename_prefix="sample_export",
        format_type="json"
    )

    print("Export results:", results)
