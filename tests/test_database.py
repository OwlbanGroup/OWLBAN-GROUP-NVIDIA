import pytest
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from jpmorgan_financial_apis.src.database_fixed import db_manager, DBBusinessModel, DBAssetModel
from jpmorgan_financial_apis.src.database_fixed import database_fixed as database_fixed_module

@pytest.fixture(scope="function")
def mock_session():
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.first.return_value = None
    mock_query.all.return_value = []
    mock_session.query.return_value = mock_query
    mock_session.add.return_value = None
    mock_session.commit.return_value = None
    mock_session.refresh.return_value = None
    mock_session.delete.return_value = None
    return mock_session

class TestDatabaseManager:
    def test_create_business(self, mock_session):
        with patch.object(database_fixed_module, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            data = {'name': 'Test Corp', 'type': 'corp'}
            business = db_manager.create_business(data)
            mock_session.add.assert_called()
            mock_session.commit.assert_called()
            assert business.name == 'Test Corp'

    def test_get_business_by_id(self, mock_session):
        with patch('jpmorgan_financial_apis.src.database_fixed.get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_query = MagicMock()
            mock_query.filter.return_value.first.return_value = DBBusinessModel(id=1, name='Test')
            mock_session.query.return_value = mock_query
            business = db_manager.get_business_by_id(1)
            assert business is not None

    def test_get_all_businesses(self, mock_session):
        with patch.object(database_fixed_module, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_query = MagicMock()
            mock_query.filter.return_value = mock_query
            mock_query.all.return_value = [DBBusinessModel(id=1), DBBusinessModel(id=2)]
            mock_session.query.return_value = mock_query
            businesses = db_manager.get_all_businesses()
            assert len(businesses) == 2

    def test_update_business(self, mock_session):
        with patch.object(database_fixed_module, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_query = MagicMock()
            mock_query.filter.return_value.first.return_value = DBBusinessModel(id=1)
            mock_session.query.return_value = mock_query
            business = db_manager.update_business(1, {'name': 'Updated'})
            mock_session.commit.assert_called()
            assert business.name == 'Updated'

    def test_delete_business(self, mock_session):
        with patch.object(database_fixed_module, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_business = DBBusinessModel(id=1)
            mock_query = MagicMock()
            mock_query.filter.return_value.first.return_value = mock_business
            mock_session.query.return_value = mock_query
            success = db_manager.delete_business(1)
            mock_get_session.assert_called_once()
            mock_session.delete.assert_called_once()
            mock_session.commit.assert_called_once()
            assert success is True

    def test_create_asset(self, mock_session):
        with patch('jpmorgan_financial_apis.src.database_fixed.get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_asset = DBAssetModel(id=1, name='Test Asset')
            mock_session.add.return_value = mock_asset
            data = {'business_id': 1, 'name': 'Test Asset', 'type': 'equipment', 'value': 10000}
            asset = db_manager.create_asset(data)
            mock_get_session.assert_called_once()
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
            assert asset.name == 'Test Asset'

    def test_get_assets_by_business_id(self, mock_session):
        with patch('jpmorgan_financial_apis.src.database_fixed.get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_assets = [DBAssetModel(id=1)]
            mock_query = MagicMock()
            mock_query.filter.return_value.all.return_value = mock_assets
            mock_session.query.return_value = mock_query
            assets = db_manager.get_assets_by_business_id(1)
            mock_get_session.assert_called_once()
            assert len(assets) == 1

    @patch('jpmorgan_financial_apis.src.database_fixed.AuditLogModel')
    def test_get_audit_logs(self, mock_audit_model, mock_session):
        with patch('jpmorgan_financial_apis.src.database_fixed.get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session
            db_manager.get_audit_logs()
            mock_get_session.assert_called_once()
            mock_session.query.assert_called()

    def test_health_check(self):
        with patch('jpmorgan_financial_apis.src.database_fixed.get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value.__enter__.return_value = mock_session
            mock_session.execute.return_value = MagicMock()
            assert db_manager.health_check() is True
            mock_get_session.assert_called_once()
            mock_session.execute.assert_called_once()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

