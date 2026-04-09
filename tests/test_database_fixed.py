import pytest
from unittest.mock import Mock, patch, MagicMock
from jpmorgan_financial_apis.src.database_fixed import (
    db_manager, async_db_manager, DBBusinessModel, DBAssetModel, AuditLogModel
)

@pytest.fixture
def mock_session():
    mock_session = MagicMock()
    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.first.return_value = None
    mock_query.all.return_value = []
    mock_session.query.return_value = mock_query
    mock_session.add.return_value = None
    mock_session.commit.return_value = None
    mock_session.execute.return_value = MagicMock()
    mock_session.delete.return_value = None
    return mock_session

class TestDatabaseManagerFixed:
    @patch('jpmorgan_financial_apis.src.database_fixed.get_session')
    def test_create_business(self, mock_get_session, mock_session):
        mock_get_session.return_value.__enter__.return_value = mock_session
        data = {'name': 'Test Corp', 'type': 'corp'}
        business = db_manager.create_business(data)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        assert business.name == 'Test Corp'

    @patch('jpmorgan_financial_apis.src.database_fixed.get_session')
    def test_get_all_businesses(self, mock_get_session, mock_session):
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_businesses = [DBBusinessModel(id=1, name='Corp1')]
        mock_session.query.return_value.all.return_value = mock_businesses
        businesses = db_manager.get_all_businesses()
        assert len(businesses) == 1
        assert businesses[0].name == 'Corp1'

    @patch('jpmorgan_financial_apis.src.database_fixed.get_session')
    def test_get_business_by_id(self, mock_get_session, mock_session):
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_business = DBBusinessModel(id=1, name='Corp1')
        mock_session.query.return_value.filter.return_value.first.return_value = mock_business
        business = db_manager.get_business_by_id(1)
        assert business.id == 1

    @patch('jpmorgan_financial_apis.src.database_fixed.get_session')
    def test_update_business(self, mock_get_session, mock_session):
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_business = DBBusinessModel(id=1, name='Old')
        mock_session.query.return_value.filter.return_value.first.return_value = mock_business
        data = {'name': 'Updated'}
        business = db_manager.update_business(1, data)
        mock_session.commit.assert_called_once()
        assert business.name == 'Updated'

    @patch('jpmorgan_financial_apis.src.database_fixed.get_session')
    def test_delete_business(self, mock_get_session, mock_session):
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_business = DBBusinessModel(id=1)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_business
        result = db_manager.delete_business(1)
        mock_session.delete.assert_called_once()
        mock_session.commit.assert_called_once()
        assert result is True

    @patch('jpmorgan_financial_apis.src.database_fixed.get_session')
    def test_create_asset(self, mock_get_session, mock_session):
        mock_get_session.return_value.__enter__.return_value = mock_session
        data = {'name': 'Stock', 'type': 'equity', 'value': 1000.0, 'business_id': 1}
        asset = db_manager.create_asset(data)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        assert asset.name == 'Stock'

    def test_get_all_assets(self):
        assets = db_manager.get_all_assets()
        assert len(assets) == 1
        assert assets[0].name == 'Test Asset'

    def test_get_asset_by_id(self):
        asset = db_manager.get_asset_by_id(1)
        assert asset is not None
        assert asset.name == 'Test Asset'

    def test_get_asset_by_id_not_found(self):
        asset = db_manager.get_asset_by_id(999)
        assert asset is None

    @patch('jpmorgan_financial_apis.src.database_fixed.get_session')
    def test_get_assets_by_business_id(self, mock_get_session, mock_session):
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_assets = [DBAssetModel(id=1, business_id=1)]
        mock_session.query.return_value.filter.return_value.all.return_value = mock_assets
        assets = db_manager.get_assets_by_business_id(1)
        assert len(assets) == 1

    def test_update_asset(self):
        asset = db_manager.update_asset(1, {'name': 'Updated'})
        assert asset.name == 'Test Asset'

    def test_delete_asset(self):
        result = db_manager.delete_asset(1)
        assert result is True
        result = db_manager.delete_asset(999)
        assert result is False

    @patch('jpmorgan_financial_apis.src.database_fixed.get_session')
    def test_health_check(self, mock_get_session, mock_session):
        mock_get_session.return_value.__enter__.return_value = mock_session
        result = db_manager.health_check()
        mock_session.execute.assert_called_once()
        assert result is True

    @patch('jpmorgan_financial_apis.src.database_fixed.get_session')
    def test_get_audit_logs(self, mock_get_session, mock_session):
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_logs = [AuditLogModel()]
        mock_session.query.return_value.all.return_value = mock_logs
        logs = db_manager.get_audit_logs()
        assert len(logs) == 1

    def test_async_db_manager_is_database_manager(self):
        from jpmorgan_financial_apis.src.database_fixed.database_fixed import AsyncDatabaseManager
        assert db_manager.__class__ == async_db_manager.__class__
        assert AsyncDatabaseManager == db_manager.__class__
