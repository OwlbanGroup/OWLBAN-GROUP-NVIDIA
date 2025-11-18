"""
Database Optimization Utilities
Provides database performance optimization tools
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Recommended database indexes for optimal performance
RECOMMENDED_INDEXES = {
    'users': [
        {'columns': ['username'], 'unique': True},
        {'columns': ['email'], 'unique': True},
        {'columns': ['created_at'], 'unique': False},
    ],
    'businesses': [
        {'columns': ['name'], 'unique': False},
        {'columns': ['registration_number'], 'unique': True},
        {'columns': ['created_at'], 'unique': False},
    ],
    'assets': [
        {'columns': ['business_id'], 'unique': False},
        {'columns': ['name'], 'unique': False},
        {'columns': ['type'], 'unique': False},
        {'columns': ['acquisition_date'], 'unique': False},
    ],
    'telemetry_events': [
        {'columns': ['timestamp'], 'unique': False},
        {'columns': ['event_type'], 'unique': False},
        {'columns': ['user_id'], 'unique': False},
    ]
}

class DatabaseOptimizer:
    """Database optimization utilities"""
    
    def __init__(self, db_session):
        """
        Initialize database optimizer
        
        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        
    def create_indexes(self, table_name: str) -> bool:
        """
        Create recommended indexes for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            bool: True if successful
        """
        try:
            if table_name not in RECOMMENDED_INDEXES:
                logger.warning(f"No index recommendations for table: {table_name}")
                return False
                
            indexes = RECOMMENDED_INDEXES[table_name]
            for index in indexes:
                columns = index['columns']
                unique = index['unique']
                index_name = f"idx_{table_name}_{'_'.join(columns)}"
                
                # Create index SQL
                unique_str = "UNIQUE" if unique else ""
                columns_str = ", ".join(columns)
                sql = f"CREATE {unique_str} INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns_str})"
                
                self.db_session.execute(sql)
                logger.info(f"Created index: {index_name}")
                
            self.db_session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error creating indexes for {table_name}: {str(e)}")
            self.db_session.rollback()
            return False
            
    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """
        Analyze query performance using EXPLAIN
        
        Args:
            query: SQL query to analyze
            
        Returns:
            dict: Query performance analysis
        """
        try:
            explain_query = f"EXPLAIN QUERY PLAN {query}"
            result = self.db_session.execute(explain_query)
            
            analysis = {
                'query': query,
                'plan': [dict(row) for row in result],
                'recommendations': []
            }
            
            # Check for table scans
            for row in analysis['plan']:
                if 'SCAN TABLE' in str(row):
                    analysis['recommendations'].append(
                        f"Consider adding index for table scan: {row}"
                    )
                    
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {'error': str(e)}
            
    def optimize_connection_pool(self, pool_size: int = 10, max_overflow: int = 20):
        """
        Configure optimal connection pool settings
        
        Args:
            pool_size: Base pool size
            max_overflow: Maximum overflow connections
        """
        # These would be set in the database engine configuration
        recommendations = {
            'pool_size': pool_size,
            'max_overflow': max_overflow,
            'pool_recycle': 3600,  # Recycle connections after 1 hour
            'pool_pre_ping': True,  # Verify connections before use
        }
        
        logger.info(f"Connection pool recommendations: {recommendations}")
        return recommendations
        
    def vacuum_database(self) -> bool:
        """
        Vacuum database to reclaim space and optimize
        
        Returns:
            bool: True if successful
        """
        try:
            self.db_session.execute("VACUUM")
            logger.info("Database vacuum completed")
            return True
        except Exception as e:
            logger.error(f"Error vacuuming database: {str(e)}")
            return False
            
    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics for a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            dict: Table statistics
        """
        try:
            # Get row count
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_result = self.db_session.execute(count_query).fetchone()
            
            # Get table size (SQLite specific)
            size_query = f"SELECT page_count * page_size as size FROM pragma_page_count('{table_name}'), pragma_page_size"
            size_result = self.db_session.execute(size_query).fetchone()
            
            stats = {
                'table_name': table_name,
                'row_count': count_result[0] if count_result else 0,
                'size_bytes': size_result[0] if size_result else 0,
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting table statistics: {str(e)}")
            return {'error': str(e)}

def apply_all_indexes(db_session) -> Dict[str, bool]:
    """
    Apply all recommended indexes to the database
    
    Args:
        db_session: SQLAlchemy database session
        
    Returns:
        dict: Results for each table
    """
    optimizer = DatabaseOptimizer(db_session)
    results = {}
    
    for table_name in RECOMMENDED_INDEXES.keys():
        results[table_name] = optimizer.create_indexes(table_name)
        
    return results

def get_optimization_report(db_session) -> Dict[str, Any]:
    """
    Generate comprehensive optimization report
    
    Args:
        db_session: SQLAlchemy database session
        
    Returns:
        dict: Optimization report
    """
    optimizer = DatabaseOptimizer(db_session)
    report = {
        'tables': {},
        'recommendations': [],
        'connection_pool': optimizer.optimize_connection_pool()
    }
    
    # Get statistics for each table
    for table_name in RECOMMENDED_INDEXES.keys():
        report['tables'][table_name] = optimizer.get_table_statistics(table_name)
        
    # Add general recommendations
    report['recommendations'].extend([
        "Enable query caching for frequently accessed data",
        "Use connection pooling with recommended settings",
        "Regularly vacuum database to reclaim space",
        "Monitor slow queries and add indexes as needed",
        "Consider read replicas for high-traffic applications"
    ])
    
    return report
