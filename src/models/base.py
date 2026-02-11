"""
Shared SQLAlchemy Base for all models
"""
from sqlalchemy.orm import registry

# Create a registry to avoid conflicts with multiple classes of same name
reg = registry()
Base = reg.generate_base()
