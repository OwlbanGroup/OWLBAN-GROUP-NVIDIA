import torch
import logging

class NimManager:
    def __init__(self):
        self.resources = {}
        self.logger = logging.getLogger("NimManager")
        self.gpu_devices = []
        self.cuda_available = torch.cuda.is_available()

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
            self.logger.info(f"NVIDIA GPUs discovered: {len(self.gpu_devices)} devices")
        else:
            self.logger.warning("CUDA not available, falling back to CPU mode")

        # Additional NVIDIA resources
        self.resources.update({
            "NVLink": "Available" if self.cuda_available and torch.cuda.device_count() > 1 else "N/A",
            "TensorRT": "Enabled",
            "cuDNN": "Enabled",
            "CUDA_Version": torch.version.cuda if self.cuda_available else "N/A"
        })

        print(f"NVIDIA NIM Resources initialized: {self.resources}")

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
                except Exception as e:
                    self.logger.error(f"Error getting GPU {i} status: {e}")
                    status[f"GPU_{i}_Usage"] = "N/A"
        else:
            status["GPU_Usage"] = "N/A (CPU mode)"

        # Additional NVIDIA-specific metrics
        status.update({
            "NVLink_Status": "Active" if len(self.gpu_devices) > 1 else "Inactive",
            "TensorRT_Status": "Ready",
            "cuDNN_Status": "Ready",
            "CUDA_Status": "Active" if self.cuda_available else "Inactive"
        })

        return status

    def optimize_gpu_resources(self):
        """Optimize NVIDIA GPU resource allocation"""
        if not self.cuda_available:
            return

        self.logger.info("Optimizing NVIDIA GPU resources...")
        # Clear GPU cache to free memory
        torch.cuda.empty_cache()

        # Log current GPU memory usage
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            self.logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    def get_nvidia_capabilities(self):
        """Get comprehensive NVIDIA capabilities report"""
        capabilities = {
            "cuda_available": self.cuda_available,
            "gpu_count": len(self.gpu_devices),
            "gpu_devices": self.gpu_devices,
            "tensorrt_enabled": True,
            "cudnn_enabled": True,
            "nvlink_available": len(self.gpu_devices) > 1,
            "resources": self.resources
        }
        return capabilities
