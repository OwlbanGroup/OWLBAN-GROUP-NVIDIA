import types
from datetime import datetime, timezone

import pytest

import src.revenue_service as rev_mod
from src.models.revenue import RevenueType, TransactionStatus


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _Ctx:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeQuery:
    def __init__(self, first_result=None, all_result=None):
        self._first = first_result
        self._all = all_result if all_result is not None else []
        self.filtered = []

    def filter_by(self, **kwargs):
        self.filtered.append(("filter_by", kwargs))
        return self

    def filter(self, *args, **kwargs):
        self.filtered.append(("filter", args, kwargs))
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def offset(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._all

    def first(self):
        return self._first


class FakeSession:
    def __init__(self, query_obj=None, query_sequence=None, commit_raises=False):
        self.query_obj = query_obj
        self.query_sequence = list(query_sequence or [])
        self.added = []
        self.committed = False
        self.refreshed = []
        self.commit_raises = commit_raises

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        if self.commit_raises:
            raise RuntimeError("commit failed")
        self.committed = True

    def refresh(self, obj):
        self.refreshed.append(obj)

    def query(self, *_args, **_kwargs):
        if self.query_sequence:
            return self.query_sequence.pop(0)
        return self.query_obj


class DummyDBManager:
    def __init__(self, session):
        self.session = session

    def get_session(self):
        return _Ctx(self.session)


@pytest.fixture(autouse=True)
def _patch_logger(monkeypatch):
    monkeypatch.setattr(rev_mod, "telemetry_logger", types.SimpleNamespace(get_logger=lambda: DummyLogger()))


def test_calculate_fees_and_taxes_purchase():
    svc = rev_mod.RevenueService()
    fee, tax = svc._calculate_fees_and_taxes(RevenueType.PURCHASE, 100.0)
    assert fee == 2.9
    assert tax == 8.0


def test_calculate_fees_and_taxes_unknown_type_fallback():
    svc = rev_mod.RevenueService()
    fee, tax = svc._calculate_fees_and_taxes(None, 100.0)  # type: ignore[arg-type]
    assert fee == 3.0
    assert tax == 8.0


def test_create_transaction_success(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(session))

    svc = rev_mod.RevenueService()
    txn = svc.create_transaction(
        user_id="u1",
        revenue_type=RevenueType.PURCHASE,
        amount=100.0,
        currency="USD",
        description="purchase",
        merchant_name="store",
        category="retail",
        payment_method="card",
        business_id=1,
        external_reference="ext-1",
    )

    assert txn.user_id == "u1"
    assert txn.status == TransactionStatus.PENDING
    assert txn.fee_amount == 2.9
    assert txn.tax_amount == 8.0
    assert txn.net_amount == 89.1
    assert len(session.added) == 1
    assert session.committed is True
    assert session.refreshed and session.refreshed[0] is txn


def test_create_transaction_raises_on_db_error(monkeypatch):
    session = FakeSession(commit_raises=True)
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(session))
    svc = rev_mod.RevenueService()

    with pytest.raises(RuntimeError):
        svc.create_transaction(user_id="u1", revenue_type=RevenueType.BILL_PAY, amount=10.0)


def test_process_transaction_not_found(monkeypatch):
    q = FakeQuery(first_result=None)
    session = FakeSession(query_obj=q)
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(session))

    svc = rev_mod.RevenueService()
    assert svc.process_transaction("missing") is False


def test_process_transaction_not_pending(monkeypatch):
    tx = types.SimpleNamespace(status=TransactionStatus.COMPLETED)
    q = FakeQuery(first_result=tx)
    session = FakeSession(query_obj=q)
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(session))

    svc = rev_mod.RevenueService()
    assert svc.process_transaction("tx1") is False


def test_process_transaction_success_and_failed_paths(monkeypatch):
    tx1 = types.SimpleNamespace(
        status=TransactionStatus.PENDING, settlement_date=None, updated_at=None
    )
    q1 = FakeQuery(first_result=tx1)
    s1 = FakeSession(query_obj=q1)
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(s1))
    svc = rev_mod.RevenueService()
    assert svc.process_transaction("tx-ok", success=True) is True
    assert tx1.status == TransactionStatus.COMPLETED
    assert isinstance(tx1.settlement_date, datetime)

    tx2 = types.SimpleNamespace(
        status=TransactionStatus.PENDING, settlement_date=None, updated_at=None
    )
    q2 = FakeQuery(first_result=tx2)
    s2 = FakeSession(query_obj=q2)
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(s2))
    svc2 = rev_mod.RevenueService()
    custom_date = datetime.now(timezone.utc)
    assert svc2.process_transaction("tx-fail", success=False, settlement_date=custom_date) is True
    assert tx2.status == TransactionStatus.FAILED
    assert tx2.settlement_date == custom_date


def test_process_transaction_exception_returns_false(monkeypatch):
    class BadSession(FakeSession):
        def query(self, *_args, **_kwargs):
            raise RuntimeError("query broken")

    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(BadSession()))
    svc = rev_mod.RevenueService()
    assert svc.process_transaction("tx") is False


def test_get_transaction_success_and_error(monkeypatch):
    tx = types.SimpleNamespace(transaction_id="tx1")
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(FakeSession(query_obj=FakeQuery(first_result=tx))))
    svc = rev_mod.RevenueService()
    assert svc.get_transaction("tx1") is tx

    class BadSession(FakeSession):
        def query(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(BadSession()))
    assert svc.get_transaction("tx1") is None


def test_get_user_transactions_success_and_error(monkeypatch):
    rows = [types.SimpleNamespace(id=1), types.SimpleNamespace(id=2)]
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(FakeSession(query_obj=FakeQuery(all_result=rows))))
    svc = rev_mod.RevenueService()
    got = svc.get_user_transactions("u1", limit=10, offset=0)
    assert len(got) == 2

    class BadSession(FakeSession):
        def query(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(BadSession()))
    assert svc.get_user_transactions("u1") == []


def test_get_revenue_metrics_with_and_without_type_and_error(monkeypatch):
    agg = types.SimpleNamespace(
        total_amount=100.0,
        total_fees=2.0,
        total_taxes=8.0,
        net_revenue=90.0,
        transaction_count=5,
        avg_transaction=20.0,
    )
    q = FakeQuery(first_result=agg)
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(FakeSession(query_obj=q)))
    svc = rev_mod.RevenueService()
    start = datetime.now(timezone.utc)
    end = datetime.now(timezone.utc)
    m = svc.get_revenue_metrics(start, end, revenue_type=RevenueType.PURCHASE)
    assert m["total_amount"] == 100.0
    assert m["transaction_count"] == 5

    agg2 = types.SimpleNamespace(
        total_amount=None, total_fees=None, total_taxes=None, net_revenue=None, transaction_count=None, avg_transaction=None
    )
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(FakeSession(query_obj=FakeQuery(first_result=agg2))))
    m2 = svc.get_revenue_metrics(start, end)
    assert m2["total_amount"] == 0.0
    assert m2["transaction_count"] == 0

    class BadSession(FakeSession):
        def query(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(BadSession()))
    assert svc.get_revenue_metrics(start, end) == {}


def test_update_daily_metrics_create_update_empty_and_error(monkeypatch):
    tx = types.SimpleNamespace(
        amount=100.0, fee_amount=2.0, tax_amount=8.0, net_amount=90.0, status=TransactionStatus.COMPLETED
    )
    metrics_q = FakeQuery(first_result=None)
    tx_q = FakeQuery(all_result=[tx])
    session = FakeSession(query_sequence=[tx_q, metrics_q] * len(list(RevenueType)))
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(session))
    svc = rev_mod.RevenueService()
    assert svc.update_daily_metrics(datetime.now(timezone.utc)) is True
    assert session.committed is True

    empty_tx_q = FakeQuery(all_result=[])
    session2 = FakeSession(query_sequence=[empty_tx_q] * len(list(RevenueType)))
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(session2))
    assert svc.update_daily_metrics(datetime.now(timezone.utc)) is True

    class BadSession(FakeSession):
        def query(self, *_args, **_kwargs):
            raise RuntimeError("broken")

    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(BadSession()))
    assert svc.update_daily_metrics(datetime.now(timezone.utc)) is False


def test_update_daily_metrics_default_date_and_existing_metric_update(monkeypatch):
    tx = types.SimpleNamespace(
        amount=50.0, fee_amount=1.0, tax_amount=4.0, net_amount=45.0, status=TransactionStatus.COMPLETED
    )

    existing_metric = types.SimpleNamespace(
        total_amount=0.0,
        total_fees=0.0,
        total_taxes=0.0,
        net_revenue=0.0,
        transaction_count=0,
        successful_transactions=0,
        failed_transactions=0,
        average_transaction_value=0.0,
        updated_at=None,
    )

    tx_q = FakeQuery(all_result=[tx])
    metrics_q = FakeQuery(first_result=existing_metric)
    session = FakeSession(query_sequence=[tx_q, metrics_q] * len(list(RevenueType)))
    monkeypatch.setattr(rev_mod, "db_manager", DummyDBManager(session))

    svc = rev_mod.RevenueService()
    assert svc.update_daily_metrics() is True
    assert existing_metric.total_amount == 50.0
    assert existing_metric.total_fees == 1.0
    assert existing_metric.total_taxes == 4.0
    assert existing_metric.net_revenue == 45.0
    assert existing_metric.transaction_count == 1
    assert existing_metric.successful_transactions == 1
    assert existing_metric.failed_transactions == 0
    assert existing_metric.average_transaction_value == 50.0
    assert existing_metric.updated_at is not None
