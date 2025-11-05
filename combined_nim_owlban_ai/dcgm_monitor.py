"""
NVIDIA DCGM Integration
Data Center GPU Manager for comprehensive GPU telemetry and health monitoring
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable

# DCGM imports - Initialize availability flag
DCGM_AVAILABLE = False
try:
    import dcgm  # type: ignore
    import dcgm_fields  # type: ignore
    import dcgm_structs  # type: ignore
    DCGM_AVAILABLE = True
except ImportError:
    dcgm = None
    dcgm_fields = None
    dcgm_structs = None

class DCGMMonitor:
    """NVIDIA DCGM-based GPU monitoring and health management"""

    def __init__(self, update_interval: float = 1.0):
        global DCGM_AVAILABLE
        self.logger = logging.getLogger("DCGMMonitor")
        self.update_interval = update_interval
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.dcgm_handle = None
        self.group_id = None
        self.gpu_count = 0

        if not DCGM_AVAILABLE:
            self.logger.warning("DCGM not available, GPU monitoring disabled")
            return

        try:
            # Initialize DCGM
            dcgm.dcgmInit()
            self.dcgm_handle = dcgm.dcgmConnect("localhost:5555")

            # Get GPU count
            self.gpu_count = dcgm.dcgmGetGpuCount(self.dcgm_handle)
            self.logger.info("DCGM initialized: monitoring %d GPUs", self.gpu_count)

            # Initialize monitoring groups
            self.group_id = dcgm.dcgmGroupCreate(self.dcgm_handle, dcgm_structs.DCGM_GROUP_DEFAULT, "owlban_group")
            for gpu_id in range(self.gpu_count):
                dcgm.dcgmGroupAddGpu(self.dcgm_handle, self.group_id, gpu_id)

            # Start field watching
            self._setup_field_watch()

        except Exception as e:
            self.logger.error("DCGM initialization failed: %s", e)
            DCGM_AVAILABLE = False

    def _setup_field_watch(self):
        """Setup DCGM field watching for comprehensive monitoring"""
        try:
            # Core GPU metrics
            fields_to_watch = [
                dcgm_fields.DCGM_FI_DEV_GPU_TEMP,           # GPU temperature
                dcgm_fields.DCGM_FI_DEV_MEMORY_TEMP,        # Memory temperature
                dcgm_fields.DCGM_FI_DEV_GPU_UTIL,           # GPU utilization
                dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,      # Memory utilization
                dcgm_fields.DCGM_FI_DEV_POWER_USAGE,        # Power consumption
                dcgm_fields.DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,  # Energy consumption
                dcgm_fields.DCGM_FI_DEV_SM_CLOCK,           # SM clock frequency
                dcgm_fields.DCGM_FI_DEV_MEM_CLOCK,          # Memory clock frequency
                dcgm_fields.DCGM_FI_DEV_PCIE_TX_BYTES,      # PCIe transmit bytes
                dcgm_fields.DCGM_FI_DEV_PCIE_RX_BYTES,      # PCIe receive bytes
                dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL,  # Single-bit ECC errors
                dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,  # Double-bit ECC errors
                dcgm_fields.DCGM_FI_DEV_XID_ERRORS,         # XID errors
                dcgm_fields.DCGM_FI_DEV_RETIRED_SBE,        # Retired single-bit errors
                dcgm_fields.DCGM_FI_DEV_RETIRED_DBE,        # Retired double-bit errors
            ]

            # Watch fields
            dcgm.dcgmWatchFields(self.dcgm_handle, self.group_id, fields_to_watch,
                               int(self.update_interval * 1000000), 3600.0, 0)

            self.logger.info("DCGM field watching configured")

        except Exception as e:
            self.logger.error("Field watch setup failed: %s", e)

    def start_monitoring(self, callback: Optional[Callable] = None):
        """Start continuous GPU monitoring"""
        if not DCGM_AVAILABLE:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(callback,))
        if self.monitor_thread:
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
        self.logger.info("DCGM monitoring started")

    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("DCGM monitoring stopped")

    def _monitor_loop(self, callback: Optional[Callable]):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                gpu_stats = self.get_gpu_stats()
                if callback:
                    callback(gpu_stats)
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error("Monitoring loop error: %s", e)
                time.sleep(self.update_interval)

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get comprehensive GPU statistics from DCGM"""
        if not DCGM_AVAILABLE:
            return {'error': 'DCGM not available'}

        try:
            # Get field values
            field_values = dcgm.dcgmGetLatestValues(self.dcgm_handle, self.group_id)

            gpu_stats = {}
            for gpu_id in range(self.gpu_count):
                gpu_key = f'gpu_{gpu_id}'
                gpu_stats[gpu_key] = self._extract_gpu_metrics(field_values, gpu_id)

            # Add system-wide statistics
            gpu_stats['system'] = self._calculate_system_stats(gpu_stats)

            return gpu_stats

        except Exception as e:
            self.logger.error("Failed to get GPU stats: %s", e)
            return {'error': str(e)}

    def _extract_gpu_metrics(self, field_values, gpu_id: int) -> Dict[str, Any]:
        """Extract metrics for a specific GPU"""
        metrics = {}

        try:
            # Temperature metrics
            metrics['gpu_temp_celsius'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_GPU_TEMP].value
            metrics['memory_temp_celsius'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_MEMORY_TEMP].value

            # Utilization metrics
            metrics['gpu_utilization_percent'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_GPU_UTIL].value
            metrics['memory_utilization_percent'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL].value

            # Power and energy metrics
            metrics['power_usage_watts'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_POWER_USAGE].value
            metrics['energy_consumption_joules'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION].value

            # Clock frequencies
            metrics['sm_clock_mhz'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_SM_CLOCK].value
            metrics['memory_clock_mhz'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_MEM_CLOCK].value

            # PCIe metrics
            metrics['pcie_tx_bytes'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_PCIE_TX_BYTES].value
            metrics['pcie_rx_bytes'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_PCIE_RX_BYTES].value

            # Error metrics
            metrics['ecc_sbe_errors'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_ECC_SBE_VOL_TOTAL].value
            metrics['ecc_dbe_errors'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_ECC_DBE_VOL_TOTAL].value
            metrics['xid_errors'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_XID_ERRORS].value
            metrics['retired_sbe'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_RETIRED_SBE].value
            metrics['retired_dbe'] = field_values[gpu_id][dcgm_fields.DCGM_FI_DEV_RETIRED_DBE].value

            # Health assessment
            metrics['health_score'] = self._calculate_gpu_health(metrics)

        except Exception as e:
            self.logger.error("Failed to extract metrics for GPU %d: %s", gpu_id, e)
            metrics['error'] = str(e)

        return metrics

    def _calculate_gpu_health(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall GPU health score (0-100)"""
        try:
            health_score = 100.0

            # Temperature penalties (max 20 points)
            gpu_temp = metrics.get('gpu_temp_celsius', 0.0)
            if gpu_temp > 80:
                health_score -= min(20, (gpu_temp - 80) * 2)

            # Utilization penalties (max 10 points)
            gpu_util = metrics.get('gpu_utilization_percent', 0.0)
            if gpu_util > 95:
                health_score -= min(10, (gpu_util - 95) * 2)

            # Error penalties (max 30 points)
            errors = (metrics.get('ecc_sbe_errors', 0) +
                     metrics.get('ecc_dbe_errors', 0) +
                     metrics.get('xid_errors', 0))
            if errors > 0:
                health_score -= min(30, errors * 10)

            # Retired page penalties (max 40 points)
            retired_pages = (metrics.get('retired_sbe', 0) +
                           metrics.get('retired_dbe', 0))
            if retired_pages > 0:
                health_score -= min(40, retired_pages * 20)

            return max(0.0, health_score)

        except Exception as e:
            self.logger.error("Health calculation failed: %s", e)
            return 50.0  # Neutral health score

    def _calculate_system_stats(self, gpu_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system-wide GPU statistics"""
        try:
            system_stats = {
                'total_gpus': self.gpu_count,
                'healthy_gpus': 0,
                'average_health_score': 0.0,
                'total_power_watts': 0.0,
                'average_gpu_temp': 0.0,
                'average_gpu_utilization': 0.0,
                'total_errors': 0,
            }

            health_scores = []
            for gpu_key, gpu_data in gpu_stats.items():
                if gpu_key.startswith('gpu_') and isinstance(gpu_data, dict):
                    health_score = gpu_data.get('health_score', 0.0)
                    health_scores.append(health_score)

                    if health_score > 70:
                        system_stats['healthy_gpus'] += 1

                    system_stats['total_power_watts'] += gpu_data.get('power_usage_watts', 0.0)
                    system_stats['average_gpu_temp'] += gpu_data.get('gpu_temp_celsius', 0.0)
                    system_stats['average_gpu_utilization'] += gpu_data.get('gpu_utilization_percent', 0.0)
                    system_stats['total_errors'] += (gpu_data.get('ecc_sbe_errors', 0) +
                                                   gpu_data.get('ecc_dbe_errors', 0) +
                                                   gpu_data.get('xid_errors', 0))
                elif gpu_key.startswith('gpu_') and isinstance(gpu_data, str):
                    # Handle error strings gracefully
                    continue

            if health_scores:
                system_stats['average_health_score'] = sum(health_scores) / len(health_scores)
                system_stats['average_gpu_temp'] /= len(health_scores)
                system_stats['average_gpu_utilization'] /= len(health_scores)

            return system_stats

        except Exception as e:
            self.logger.error("System stats calculation failed: %s", e)
            return {'error': str(e)}

    def get_health_alerts(self) -> List[Dict[str, Any]]:
        """Get current health alerts for GPUs"""
        if not DCGM_AVAILABLE:
            return []

        try:
            gpu_stats = self.get_gpu_stats()
            alerts = []

            for gpu_key, gpu_data in gpu_stats.items():
                if gpu_key.startswith('gpu_') and isinstance(gpu_data, dict):
                    gpu_id = int(gpu_key.split('_')[1])

                    # Temperature alerts
                    if isinstance(gpu_data, dict):
                        gpu_temp = gpu_data.get('gpu_temp_celsius', 0.0)
                        if gpu_temp > 85:
                            alerts.append({
                                'gpu_id': gpu_id,
                                'severity': 'critical',
                                'type': 'temperature',
                                'message': f'GPU {gpu_id} temperature critically high: {gpu_temp}°C',
                                'value': gpu_temp
                            })
                        elif gpu_temp > 75:
                            alerts.append({
                                'gpu_id': gpu_id,
                                'severity': 'warning',
                                'type': 'temperature',
                                'message': f'GPU {gpu_id} temperature elevated: {gpu_temp}°C',
                                'value': gpu_temp
                            })

                        # Error alerts
                        errors = (gpu_data.get('ecc_dbe_errors', 0) +
                                 gpu_data.get('xid_errors', 0))
                        if errors > 0:
                            alerts.append({
                                'gpu_id': gpu_id,
                                'severity': 'critical',
                                'type': 'errors',
                                'message': f'GPU {gpu_id} has {errors} critical errors',
                                'value': errors
                            })

                        # Health alerts
                        health_score = gpu_data.get('health_score', 100.0)
                        if health_score < 50:
                            alerts.append({
                                'gpu_id': gpu_id,
                                'severity': 'critical',
                                'type': 'health',
                                'message': f'GPU {gpu_id} health critically low: {health_score}',
                                'value': health_score
                            })
                        elif health_score < 70:
                            alerts.append({
                                'gpu_id': gpu_id,
                                'severity': 'warning',
                                'type': 'health',
                                'message': f'GPU {gpu_id} health degraded: {health_score}',
                                'value': health_score
                            })

            return alerts

        except Exception as e:
            self.logger.error("Health alerts retrieval failed: %s", e)
            return []

    def __del__(self):
        """Cleanup DCGM resources"""
        if DCGM_AVAILABLE:
            try:
                dcgm.dcgmDisconnect(self.dcgm_handle)
                dcgm.dcgmShutdown()
            except Exception:
                pass
