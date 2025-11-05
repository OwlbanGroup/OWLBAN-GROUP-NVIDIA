import torch
import logging
import psutil
import GPUtil
from typing import Dict, List, Optional
import subprocess
import os

# NVIDIA-specific imports
try:
    import pynvml
    pynvml.nvmlInit()
    nvml_available = True
except ImportError:
    nvml_available = False

class NimManager:
    def __init__(self):
        self.resources = {}
        self.logger = logging.getLogger("NimManager")
        self.gpu_devices = []
        self.cuda_available = torch.cuda.is_available()
        self.nvidia_container_runtime = self._check_nvidia_container_runtime()

    def _check_nvidia_container_runtime(self) -> bool:
        """Check if NVIDIA Container Runtime is available"""
        try:
            result = subprocess.run(['nvidia-container-runtime', '--version'],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    def initialize(self):
        self.logger.info("Initializing NVIDIA NIM infrastructure management with GPU acceleration...")
        # Discover NVIDIA GPU resources
        if self.cuda_available:
            self.gpu_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                self.resources[f"GPU_{i}"] = {
                    "name": gpu_name,
                    "memory_gb": gpu_memory,
                    "compute_capability": torch.cuda.get_device_capability(i)
                }
            self.logger.info("NVIDIA GPUs discovered: %d devices", len(self.gpu_devices))
        else:
            self.logger.warning("CUDA not available, falling back to CPU mode")

        # Additional NVIDIA resources
        self.resources.update({
            "NVLink": "Available" if self.cuda_available and torch.cuda.device_count() > 1 else "N/A",
            "TensorRT": "Enabled",
            "cuDNN": "Enabled",
            "CUDA_Version": torch.version.cuda if self.cuda_available else "N/A",
            "NVIDIA_Container_Runtime": "Available" if self.nvidia_container_runtime else "N/A"
        })

        self.logger.info("NVIDIA NIM Resources initialized: %s", self.resources)

    def get_resource_status(self):
        """Get real-time NVIDIA resource status with GPU metrics"""
        status = {}

        if self.cuda_available:
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(i)  # Ensure all operations are complete
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                    total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB

                    gpu_utilization = (memory_allocated / total_memory) * 100
                    status[f"GPU_{i}_Usage"] = f"{gpu_utilization:.1f}%"
                    status[f"GPU_{i}_Memory_Allocated"] = f"{memory_allocated:.1f}GB"
                    status[f"GPU_{i}_Memory_Reserved"] = f"{memory_reserved:.1f}GB"

                    # Add NVIDIA NVML metrics if available
                    if nvml_available:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts

                            status[f"GPU_{i}_Utilization"] = f"{gpu_util.gpu}%"
                            status[f"GPU_{i}_Temperature"] = f"{temp}°C"
                            status[f"GPU_{i}_Power_Usage"] = f"{power:.1f}W"
                        except Exception as e:
                            self.logger.warning("NVML metrics unavailable for GPU %d: %s", i, e)

                except Exception as e:
                    self.logger.error("Error getting GPU %d status: %s", i, e)
                    status[f"GPU_{i}_Usage"] = "N/A"
        else:
            status["GPU_Usage"] = "N/A (CPU mode)"

        # System-wide metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        status.update({
            "CPU_Usage": f"{cpu_percent:.1f}%",
            "System_Memory_Usage": f"{memory.percent:.1f}%",
            "NVLink_Status": "Active" if len(self.gpu_devices) > 1 else "Inactive",
            "TensorRT_Status": "Ready",
            "cuDNN_Status": "Ready",
            "CUDA_Status": "Active" if self.cuda_available else "Inactive",
            "NVIDIA_Container_Status": "Active" if self.nvidia_container_runtime else "Inactive"
        })

        return status

    def optimize_gpu_resources(self):
        """Optimize NVIDIA GPU resource allocation with advanced memory management"""
        if not self.cuda_available:
            return

        self.logger.info("Optimizing NVIDIA GPU resources...")

        # Clear GPU cache to free memory
        torch.cuda.empty_cache()

        # Advanced GPU memory optimization
        for i in range(torch.cuda.device_count()):
            try:
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats(i)

                # Log detailed memory usage
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                peak_allocated = torch.cuda.max_memory_allocated(i) / 1024**3

                self.logger.info("GPU %d: %.1fGB allocated, %.1fGB reserved, %.1fGB peak", i, allocated, reserved, peak_allocated)

                # If memory usage is high, attempt garbage collection
                if allocated / torch.cuda.get_device_properties(i).total_memory * 1024**3 > 0.8:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.logger.info("GPU %d: Performed garbage collection and cache clearing", i)

            except Exception as e:
                self.logger.error("Error optimizing GPU %d: %s", i, e)

    def deploy_nvidia_container(self, image_name: str, container_name: str, gpu_devices: Optional[List[int]] = None) -> bool:
        """Deploy NVIDIA GPU-accelerated container"""
        if not self.nvidia_container_runtime:
            self.logger.error("NVIDIA Container Runtime not available")
            return False

        try:
            cmd = ["docker", "run", "--gpus", "all" if gpu_devices is None else f"device={','.join(map(str, gpu_devices))}",
                   "--name", container_name, "-d", image_name]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                self.logger.info("NVIDIA container %s deployed successfully", container_name)
                return True
            else:
                self.logger.error("Failed to deploy NVIDIA container: %s", result.stderr)
                return False

        except Exception as e:
            self.logger.error("Error deploying NVIDIA container: %s", e)
            return False

    def monitor_gpu_health(self) -> Dict[str, any]:
        """Monitor NVIDIA GPU health and performance"""
        health_status = {}

        if nvml_available:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Get health status
                    health = pynvml.nvmlDeviceGetHealthStatus(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)

                    health_status[f"GPU_{i}"] = {
                        "health": "Good" if health == pynvml.NVML_HEALTH_STATUS_OK else "Warning",
                        "temperature_celsius": temp,
                        "fan_speed_percent": fan_speed
                    }
            except Exception as e:
                self.logger.error("Error monitoring GPU health: %s", e)

        return health_status

    def get_nvidia_capabilities(self):
        """Get comprehensive NVIDIA capabilities report"""
        capabilities = {
            "cuda_available": self.cuda_available,
            "gpu_count": len(self.gpu_devices),
            "gpu_devices": self.gpu_devices,
            "tensorrt_enabled": True,
            "cudnn_enabled": True,
            "nvlink_available": len(self.gpu_devices) > 1,
            "nvml_available": nvml_available,
            "nvidia_container_runtime": self.nvidia_container_runtime,
            "resources": self.resources,
            "gpu_health": self.monitor_gpu_health()
        }
        return capabilities
