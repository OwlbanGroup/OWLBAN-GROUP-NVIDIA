#!/usr/bin/env python3
"""
Production Deployment Script for JPMorgan Financial APIs
Handles complete production deployment with health checks and monitoring
"""
import os
import sys
import time
import logging
import subprocess
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/deploy.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ProductionDeployer:
    """Production deployment manager"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.server_process = None
        self.health_check_url = "http://localhost:8000/health"

    def check_prerequisites(self):
        """Check all prerequisites for production deployment"""
        logger.info("🔍 Checking deployment prerequisites...")

        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8+ required for production deployment")

        # Check required packages
        required_packages = [
            'flask', 'waitress', 'flask_limiter', 'flask_cors',
            'flask_restx', 'flask_talisman', 'prometheus_client',
            'werkzeug', 'numpy', 'redis', 'dotenv'
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.error(f"❌ Missing required packages: {', '.join(missing_packages)}")
            logger.info("Installing missing packages...")
            for package in missing_packages:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

        # Check environment variables
        required_env_vars = ['SECRET_KEY']
        missing_env_vars = []
        for var in required_env_vars:
            if not os.environ.get(var):
                missing_env_vars.append(var)

        if missing_env_vars:
            logger.warning(f"⚠️ Missing environment variables: {', '.join(missing_env_vars)}")
            logger.info("Setting default values for missing environment variables...")
            os.environ.setdefault('SECRET_KEY', 'production-secret-key-change-in-env')

        # Create necessary directories
        dirs_to_create = ['logs', 'backups', 'temp']
        for dir_name in dirs_to_create:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✅ Created directory: {dir_path}")

        logger.info("✅ All prerequisites checked successfully")

    def backup_current_deployment(self):
        """Create backup of current deployment"""
        logger.info("💾 Creating deployment backup...")

        backup_dir = self.project_root / 'backups'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f'backup_{timestamp}'

        try:
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup important files
            files_to_backup = [
                'app_final.py',
                'production_server.py',
                'requirements.txt',
                'config.py',
                '.env',
                'telemetry.db'
            ]

            for file_name in files_to_backup:
                src = self.project_root / file_name
                if src.exists():
                    dst = backup_path / file_name
                    dst.write_bytes(src.read_bytes())
                    logger.info(f"✅ Backed up: {file_name}")

            logger.info(f"✅ Backup created at: {backup_path}")

        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise

    def start_production_server(self):
        """Start the production server"""
        logger.info("🚀 Starting production server...")

        try:
            # Start server in background
            self.server_process = subprocess.Popen([
                sys.executable, 'production_server.py'
            ], cwd=self.project_root)

            # Wait for server to start
            logger.info("⏳ Waiting for server to start...")
            time.sleep(5)

            # Check if process is still running
            if self.server_process.poll() is not None:
                raise RuntimeError("Server process exited immediately")

            logger.info("✅ Production server started successfully")

        except Exception as e:
            logger.error(f"❌ Failed to start production server: {e}")
            raise

    def perform_health_checks(self):
        """Perform comprehensive health checks"""
        logger.info("🏥 Performing health checks...")

        # Wait for server to be ready
        max_retries = 30
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.get(self.health_check_url, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Health check passed")
                    break
                else:
                    logger.warning(f"Health check returned status: {response.status_code}")
            except requests.RequestException as e:
                logger.warning(f"Health check failed (attempt {retry_count + 1}/{max_retries}): {e}")

            retry_count += 1
            time.sleep(2)

        if retry_count >= max_retries:
            raise RuntimeError("Health checks failed - server may not be running properly")

        # Additional endpoint checks
        endpoints_to_check = [
            ('/', 'Root endpoint'),
            ('/dashboard', 'Dashboard endpoint'),
            ('/telemetry/metrics?hours=1', 'Metrics endpoint'),
        ]

        for endpoint, description in endpoints_to_check:
            try:
                url = f"http://localhost:8000{endpoint}"
                response = requests.get(url, timeout=10)
                if response.status_code in [200, 302]:  # 302 for dashboard redirect
                    logger.info(f"✅ {description} accessible")
                else:
                    logger.warning(f"⚠️ {description} returned status: {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ {description} check failed: {e}")

        logger.info("✅ Health checks completed")

    def monitor_deployment(self):
        """Monitor the deployment for a short period"""
        logger.info("📊 Monitoring deployment...")

        # Monitor for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            try:
                response = requests.get(self.health_check_url, timeout=5)
                if response.status_code != 200:
                    logger.warning(f"⚠️ Health check failed during monitoring: {response.status_code}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"❌ Monitoring failed: {e}")
                break

        logger.info("✅ Monitoring completed")

    def deploy(self):
        """Execute complete production deployment"""
        logger.info("🏭 Starting production deployment...")

        try:
            # Step 1: Prerequisites
            self.check_prerequisites()

            # Step 2: Backup
            self.backup_current_deployment()

            # Step 3: Start server
            self.start_production_server()

            # Step 4: Health checks
            self.perform_health_checks()

            # Step 5: Monitor
            self.monitor_deployment()

            logger.info("🎉 Production deployment completed successfully!")
            logger.info("📍 Server is running at: http://localhost:8000")
            logger.info("🔍 Check logs/deploy.log for detailed information")

            return True

        except Exception as e:
            logger.error(f"❌ Production deployment failed: {e}")
            self.rollback()
            return False

    def rollback(self):
        """Rollback deployment in case of failure"""
        logger.info("🔄 Rolling back deployment...")

        # Stop server if running
        if self.server_process and self.server_process.poll() is None:
            logger.info("Stopping server process...")
            self.server_process.terminate()
            self.server_process.wait(timeout=10)

        # Restore from latest backup
        backup_dir = self.project_root / 'backups'
        if backup_dir.exists():
            backups = sorted(backup_dir.iterdir(), reverse=True)
            if backups:
                latest_backup = backups[0]
                logger.info(f"Restoring from backup: {latest_backup}")

                files_to_restore = [
                    'app_final.py',
                    'production_server.py',
                    'requirements.txt',
                    'config.py',
                    '.env'
                ]

                for file_name in files_to_restore:
                    backup_file = latest_backup / file_name
                    if backup_file.exists():
                        target_file = self.project_root / file_name
                        target_file.write_bytes(backup_file.read_bytes())
                        logger.info(f"✅ Restored: {file_name}")

        logger.info("✅ Rollback completed")

def main():
    """Main deployment function"""
    deployer = ProductionDeployer()

    success = deployer.deploy()

    if success:
        print("\n🎉 DEPLOYMENT SUCCESSFUL!")
        print("📍 Your JPMorgan Financial APIs are now running in production!")
        print("🌐 Access the application at: http://localhost:8000")
        print("📊 Check logs/deploy.log for deployment details")
        return 0
    else:
        print("\n❌ DEPLOYMENT FAILED!")
        print("📋 Check logs/deploy.log for error details")
        return 1

if __name__ == "__main__":
    exit(main())
