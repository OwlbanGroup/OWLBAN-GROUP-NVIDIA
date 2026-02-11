#!/usr/bin/env python3
"""
Performance Optimization Script
Applies database indexes, runs performance analysis, and optimizes settings
"""
import sys
import os
from datetime import datetime, timezone

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from src.database_optimizer import DatabaseOptimizer, apply_all_indexes, get_optimization_report
    from src.database_fixed import db_manager
    from src.logger import telemetry_logger
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def apply_database_indexes():
    """Apply all recommended database indexes"""
    logger = telemetry_logger.get_logger()
    logger.info("Applying database indexes...")

    try:
        # Get database session
        session = db_manager.get_session()

        # Apply all indexes
        results = apply_all_indexes(session)

        # Log results
        success_count = sum(1 for result in results.values() if result)
        total_count = len(results)

        print(f"Database indexes applied: {success_count}/{total_count} successful")

        for table, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {status} {table}")

        return True

    except Exception as e:
        print(f"ERROR: Error applying indexes: {e}")
        return False
    finally:
        if 'session' in locals():
            session.close()

def run_query_performance_analysis():
    """Run performance analysis on key queries"""
    print("Running query performance analysis...")

    try:
        session = db_manager.get_session()
        optimizer = DatabaseOptimizer(session)

        # Key queries to analyze
        key_queries = [
            "SELECT * FROM users WHERE username = ?",
            "SELECT * FROM businesses WHERE name LIKE ?",
            "SELECT * FROM assets WHERE business_id = ?",
            "SELECT COUNT(*) FROM telemetry_events WHERE timestamp > ?",
            "SELECT SUM(amount) FROM revenue_transactions WHERE user_id = ? AND status = 'completed'"
        ]

        analysis_results = {}
        for query in key_queries:
            try:
                analysis = optimizer.analyze_query_performance(query)
                analysis_results[query] = analysis
                print(f"  Analyzed: {query[:50]}...")
            except Exception as e:
                print(f"  Failed to analyze: {query[:50]}... - {e}")
                analysis_results[query] = {'error': str(e)}

        print(f"Query analysis completed for {len(analysis_results)} queries")
        return analysis_results

    except Exception as e:
        print(f"ERROR: Error in query analysis: {e}")
        return {}
    finally:
        if 'session' in locals():
            session.close()

def optimize_connection_pool():
    """Optimize database connection pool settings"""
    print("Optimizing connection pool...")

    try:
        session = db_manager.get_session()
        optimizer = DatabaseOptimizer(session)

        recommendations = optimizer.optimize_connection_pool(
            pool_size=10,
            max_overflow=20
        )

        print("Connection pool optimization recommendations:")
        for key, value in recommendations.items():
            print(f"  {key}: {value}")

        return recommendations

    except Exception as e:
        print(f"ERROR: Error optimizing connection pool: {e}")
        return {}
    finally:
        if 'session' in locals():
            session.close()

def generate_optimization_report():
    """Generate comprehensive optimization report"""
    print("Generating optimization report...")

    try:
        session = db_manager.get_session()
        report = get_optimization_report(session)

        print("Optimization report generated:")
        print(f"  Tables analyzed: {len(report.get('tables', {}))}")
        print(f"  Recommendations: {len(report.get('recommendations', []))}")

        # Print table statistics
        print("  Table Statistics:")
        for table_name, stats in report.get('tables', {}).items():
            print(f"    {table_name}: {stats.get('row_count', 0)} rows, {stats.get('size_bytes', 0)} bytes")

        # Print recommendations
        print("  Recommendations:")
        for rec in report.get('recommendations', []):
            print(f"    • {rec}")

        return report

    except Exception as e:
        print(f"ERROR: Error generating report: {e}")
        return {}
    finally:
        if 'session' in locals():
            session.close()

def main():
    """Main performance optimization function"""
    print("Starting Performance Optimization...")
    print(f"Start time: {datetime.now(timezone.utc).isoformat()}")

    results = {
        'indexes_applied': False,
        'query_analysis': {},
        'connection_pool': {},
        'optimization_report': {},
        'timestamp': datetime.now(timezone.utc).isoformat()
    }

    # Step 1: Apply database indexes
    results['indexes_applied'] = apply_database_indexes()

    # Step 2: Run query performance analysis
    results['query_analysis'] = run_query_performance_analysis()

    # Step 3: Optimize connection pool
    results['connection_pool'] = optimize_connection_pool()

    # Step 4: Generate optimization report
    results['optimization_report'] = generate_optimization_report()

    print("\nPerformance optimization completed!")
    print(f"End time: {datetime.now(timezone.utc).isoformat()}")

    # Log completion
    logger = telemetry_logger.get_logger()
    logger.info("Performance optimization completed", extra={
        'optimization_results': results
    })

    return results

if __name__ == '__main__':
    main()
