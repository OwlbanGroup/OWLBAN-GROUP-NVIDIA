"""
Batch Models for JPMorgan Financial APIs
Defines data models for batch processing runs.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class BatchStatus(str, Enum):
    """Batch status enumeration"""
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class BatchRun(Base):
    """
    Batch run model for tracking batch processing operations
    """
    __tablename__ = 'batch_runs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(36), nullable=False, unique=True)
    batch_type = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default=BatchStatus.STARTED.value)
    start_time = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    end_time = Column(DateTime(timezone=True), nullable=True)
    batch_size = Column(Integer, nullable=False)
    priority = Column(String(20), default="normal")
    processed_count = Column(Integer, default=0)
    total_count = Column(Integer, nullable=False)
    metadata = Column(JSON, default=dict)

    def __init__(self, batch_id: str, batch_type: str, batch_size: int, total_count: int,
                 priority: str = "normal", status: BatchStatus = BatchStatus.STARTED,
                 start_time: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None):
        self.batch_id = batch_id
        self.batch_type = batch_type
        self.status = status.value
        self.start_time = start_time or datetime.now(timezone.utc)
        self.batch_size = batch_size
        self.priority = priority
        self.processed_count = 0
        self.total_count = total_count
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert batch run object to dictionary

        Returns:
            Dict representation of the batch run
        """
        return {
            'id': self.id,
            'batch_id': self.batch_id,
            'batch_type': self.batch_type,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'batch_size': self.batch_size,
            'priority': self.priority,
            'processed_count': self.processed_count,
            'total_count': self.total_count,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchRun':
        """
        Create batch run object from dictionary

        Args:
            data: Dictionary containing batch run data

        Returns:
            BatchRun object
        """
        return cls(
            batch_id=data['batch_id'],
            batch_type=data['batch_type'],
            batch_size=data['batch_size'],
            total_count=data['total_count'],
            priority=data.get('priority', 'normal'),
            status=BatchStatus(data['status']),
            start_time=datetime.fromisoformat(data['start_time']) if data.get('start_time') else None,
            metadata=data.get('metadata', {})
        )
