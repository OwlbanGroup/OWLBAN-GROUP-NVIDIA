"""
NVIDIA GPU Telemetry Parser for processing GPU-specific telemetry data
"""
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import re

from .logger import telemetry_logger

@dataclass
class GPUTelemetryEvent:
    """Data class representing a parsed NVIDIA GPU telemetry event"""
    timestamp: str
    gpu_id: int
    gpu_name: str
    driver_version: str
    cuda_version: str
    gpu_utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_utilization_percent: float
    temperature_celsius: float
    power_draw_watts: float
    power_limit_watts: float
    fan_speed_percent: float
    clock_graphics_mhz: int
    clock_sm_mhz: int
    clock_memory_mhz: int
    clock_video_mhz: int
    pcie_link_gen: int
    pcie_link_width: int
    processes: List[Dict[str, Any]]
    event_type: str
    hostname: str
    nvidia_smi_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

class NVIDIATelemetryParser:
    """Parser for NVIDIA GPU telemetry data from nvidia-smi and DCGM"""

    def __init__(self):
        self.logger = telemetry_logger

    def parse_nvidia_smi_output(self, raw_output: str, timestamp: Optional[str] = None) -> Optional[GPUTelemetryEvent]:
        """
        Parse raw nvidia-smi output into a structured GPUTelemetryEvent

        Args:
            raw_output: Raw nvidia-smi command output
            timestamp: Optional timestamp, defaults to current time

        Returns:
            GPUTelemetryEvent object if parsing successful, None otherwise
        """
        try:
            if timestamp is None:
                timestamp = datetime.utcnow().isoformat()

            # Extract GPU information using regex patterns
            gpu_info = self._extract_gpu_info(raw_output)
            if not gpu_info:
                return None

            # Extract processes information
            processes = self._extract_processes(raw_output)

            # Extract driver and CUDA versions
            driver_version = self._extract_driver_version(raw_output)
            cuda_version = self._extract_cuda_version(raw_output)
            nvidia_smi_version = self._extract_nvidia_smi_version(raw_output)

            event = GPUTelemetryEvent(
                timestamp=timestamp,
                gpu_id=gpu_info.get('gpu_id', 0),
                gpu_name=gpu_info.get('gpu_name', 'Unknown'),
                driver_version=driver_version,
                cuda_version=cuda_version,
                gpu_utilization_percent=gpu_info.get('gpu_utilization', 0.0),
                memory_used_mb=gpu_info.get('memory_used', 0.0),
                memory_total_mb=gpu_info.get('memory_total', 0.0),
                memory_utilization_percent=gpu_info.get('memory_utilization', 0.0),
                temperature_celsius=gpu_info.get('temperature', 0.0),
                power_draw_watts=gpu_info.get('power_draw', 0.0),
                power_limit_watts=gpu_info.get('power_limit', 0.0),
                fan_speed_percent=gpu_info.get('fan_speed', 0.0),
                clock_graphics_mhz=gpu_info.get('clock_graphics', 0),
                clock_sm_mhz=gpu_info.get('clock_sm', 0),
                clock_memory_mhz=gpu_info.get('clock_memory', 0),
                clock_video_mhz=gpu_info.get('clock_video', 0),
                pcie_link_gen=gpu_info.get('pcie_link_gen', 0),
                pcie_link_width=gpu_info.get('pcie_link_width', 0),
                processes=processes,
                event_type='nvidia_smi',
                hostname=self._get_hostname(),
                nvidia_smi_version=nvidia_smi_version
            )

            # Log the parsed event
            self.logger.get_logger().info(f"Parsed GPU telemetry for GPU {gpu_info.get('gpu_id', 0)}")

            return event

        except Exception as e:
            self.logger.log_error(e, {'context': 'parsing_nvidia_smi_output'})
            return None

    def parse_dcgm_metrics(self, metrics_data: Dict[str, Any]) -> Optional[GPUTelemetryEvent]:
        """
        Parse DCGM (Data Center GPU Manager) metrics into GPUTelemetryEvent

        Args:
            metrics_data: DCGM metrics data dictionary

        Returns:
            GPUTelemetryEvent object if parsing successful, None otherwise
        """
        try:
            timestamp = metrics_data.get('timestamp', datetime.utcnow().isoformat())
            gpu_id = metrics_data.get('gpu_id', 0)

            event = GPUTelemetryEvent(
                timestamp=timestamp,
                gpu_id=gpu_id,
                gpu_name=metrics_data.get('gpu_name', 'Unknown'),
                driver_version=metrics_data.get('driver_version', ''),
                cuda_version=metrics_data.get('cuda_version', ''),
                gpu_utilization_percent=metrics_data.get('gpu_utilization', 0.0),
                memory_used_mb=metrics_data.get('memory_used_mb', 0.0),
                memory_total_mb=metrics_data.get('memory_total_mb', 0.0),
                memory_utilization_percent=metrics_data.get('memory_utilization', 0.0),
                temperature_celsius=metrics_data.get('temperature_celsius', 0.0),
                power_draw_watts=metrics_data.get('power_draw_watts', 0.0),
                power_limit_watts=metrics_data.get('power_limit_watts', 0.0),
                fan_speed_percent=metrics_data.get('fan_speed_percent', 0.0),
                clock_graphics_mhz=metrics_data.get('clock_graphics_mhz', 0),
                clock_sm_mhz=metrics_data.get('clock_sm_mhz', 0),
                clock_memory_mhz=metrics_data.get('clock_memory_mhz', 0),
                clock_video_mhz=metrics_data.get('clock_video_mhz', 0),
                pcie_link_gen=metrics_data.get('pcie_link_gen', 0),
                pcie_link_width=metrics_data.get('pcie_link_width', 0),
                processes=metrics_data.get('processes', []),
                event_type='dcgm',
                hostname=metrics_data.get('hostname', self._get_hostname()),
                nvidia_smi_version=metrics_data.get('nvidia_smi_version', '')
            )

            return event

        except Exception as e:
            self.logger.log_error(e, {'context': 'parsing_dcgm_metrics'})
            return None

    def _extract_gpu_info(self, nvidia_smi_output: str) -> Optional[Dict[str, Any]]:
        """Extract GPU information from nvidia-smi output"""
        try:
            # GPU ID - more robust pattern
            gpu_id_match = re.search(r'GPU\s+(\d+)\s*:', nvidia_smi_output)
            gpu_id = int(gpu_id_match.group(1)) if gpu_id_match else 0

            # GPU Name - improved pattern to handle various formats including multi-word names
            name_match = re.search(r'GPU\s+\d+\s*:\s*([^\(\n]+?)(?:\s*\(|\s*$|\s*\n)', nvidia_smi_output)
            if not name_match:
                # Fallback pattern for different formats
                name_match = re.search(r'(\w+(?:\s+\w+)*)\s*\(UUID:', nvidia_smi_output)
            gpu_name = name_match.group(1).strip() if name_match else 'Unknown'

            # GPU Utilization - more specific pattern
            util_match = re.search(r'(\d+)%\s+Default\s+', nvidia_smi_output)
            gpu_utilization = float(util_match.group(1)) if util_match else 0.0

            # Memory Usage - improved pattern for different formats
            mem_match = re.search(r'(\d+(?:\.\d+)?)\s*MiB\s*/\s*(\d+(?:\.\d+)?)\s*MiB', nvidia_smi_output)
            if mem_match:
                memory_used = float(mem_match.group(1))
                memory_total = float(mem_match.group(2))
                memory_utilization = (memory_used / memory_total) * 100 if memory_total > 0 else 0.0
            else:
                memory_used = memory_total = memory_utilization = 0.0

            # Temperature - more specific pattern
            temp_match = re.search(r'(\d+(?:\.\d+)?)C\s+', nvidia_smi_output)
            temperature = float(temp_match.group(1)) if temp_match else 0.0

            # Power - improved pattern
            power_match = re.search(r'(\d+(?:\.\d+)?)W\s*/\s*(\d+(?:\.\d+)?)W', nvidia_smi_output)
            if power_match:
                power_draw = float(power_match.group(1))
                power_limit = float(power_match.group(2))
            else:
                power_draw = power_limit = 0.0

            # Fan Speed - more specific pattern for fan speed
            fan_match = re.search(r'(\d+)%\s+(?:\d+)%\s+(\d+)%', nvidia_smi_output)
            fan_speed = float(fan_match.group(1)) if fan_match else 0.0

            # Validate extracted values
            if memory_used < 0 or memory_total < 0:
                self.logger.log_error(ValueError("Invalid memory values extracted"), {'context': 'gpu_info_validation'})
                return None
            if temperature < 0 or temperature > 150:
                self.logger.log_error(ValueError("Invalid temperature value extracted"), {'context': 'gpu_info_validation'})
                return None
            if power_draw < 0 or power_limit < 0:
                self.logger.log_error(ValueError("Invalid power values extracted"), {'context': 'gpu_info_validation'})
                return None

            # Clocks
            clocks = self._extract_clocks(nvidia_smi_output)

            # PCIe
            pcie_info = self._extract_pcie_info(nvidia_smi_output)

            return {
                'gpu_id': gpu_id,
                'gpu_name': gpu_name,
                'gpu_utilization': gpu_utilization,
                'memory_used': memory_used,
                'memory_total': memory_total,
                'memory_utilization': memory_utilization,
                'temperature': temperature,
                'power_draw': power_draw,
                'power_limit': power_limit,
                'fan_speed': fan_speed,
                **clocks,
                **pcie_info
            }

        except Exception as e:
            self.logger.log_error(e, {'context': 'extracting_gpu_info'})
            return None

    def _extract_processes(self, nvidia_smi_output: str) -> List[Dict[str, Any]]:
        """Extract process information from nvidia-smi output"""
        processes = []

        try:
            # Find the processes section
            process_section = re.search(r'Processes:(.*?)(?:\n\n|\Z)', nvidia_smi_output, re.DOTALL)
            if not process_section:
                return processes

            # Parse each process line
            lines = process_section.group(1).strip().split('\n')
            for line in lines[1:]:  # Skip header
                if line.strip():
                    # Extract PID, process name, and memory usage
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            pid = int(parts[1])
                            memory_mb = float(parts[3].replace('MiB', ''))
                            process_name = ' '.join(parts[4:]) if len(parts) > 4 else 'Unknown'

                            processes.append({
                                'pid': pid,
                                'name': process_name,
                                'memory_mb': memory_mb
                            })
                        except (ValueError, IndexError):
                            continue

        except Exception as e:
            self.logger.log_error(e, {'context': 'extracting_processes'})

        return processes

    def _extract_driver_version(self, nvidia_smi_output: str) -> str:
        """Extract driver version"""
        match = re.search(r'Driver Version:\s*([^\s]+)', nvidia_smi_output)
        return match.group(1) if match else ''

    def _extract_cuda_version(self, nvidia_smi_output: str) -> str:
        """Extract CUDA version"""
        match = re.search(r'CUDA Version:\s*([^\s]+)', nvidia_smi_output)
        return match.group(1) if match else ''

    def _extract_nvidia_smi_version(self, nvidia_smi_output: str) -> str:
        """Extract NVIDIA SMI version"""
        match = re.search(r'NVIDIA-SMI\s*([^\s]+)', nvidia_smi_output)
        return match.group(1) if match else ''

    def _extract_clocks(self, nvidia_smi_output: str) -> Dict[str, int]:
        """Extract clock frequencies"""
        clocks = {
            'clock_graphics': 0,
            'clock_sm': 0,
            'clock_memory': 0,
            'clock_video': 0
        }

        try:
            # Find the Clocks section and extract all clock information
            clocks_section = re.search(r'Clocks\s*\n(.*?)(?:\n\s*\n|\Z)', nvidia_smi_output, re.DOTALL)
            if clocks_section:
                section_content = clocks_section.group(1)

                # Graphics clock - look for Graphics pattern
                graphics_match = re.search(r'Graphics\s*:\s*(\d+(?:\.\d+)?)\s*MHz', section_content, re.IGNORECASE)
                if graphics_match:
                    clocks['clock_graphics'] = int(float(graphics_match.group(1)))

                # SM clock (streaming multiprocessor)
                sm_match = re.search(r'SM\s*:\s*(\d+(?:\.\d+)?)\s*MHz', section_content, re.IGNORECASE)
                if sm_match:
                    clocks['clock_sm'] = int(float(sm_match.group(1)))

                # Memory clock
                mem_match = re.search(r'Memory\s*:\s*(\d+(?:\.\d+)?)\s*MHz', section_content, re.IGNORECASE)
                if mem_match:
                    clocks['clock_memory'] = int(float(mem_match.group(1)))

                # Video clock
                video_match = re.search(r'Video\s*:\s*(\d+(?:\.\d+)?)\s*MHz', section_content, re.IGNORECASE)
                if video_match:
                    clocks['clock_video'] = int(float(video_match.group(1)))

            # Fallback: if no Clocks section found, try to extract from anywhere in output
            if all(v == 0 for v in clocks.values()):
                # Graphics clock fallback
                graphics_match = re.search(r'Graphics\s*:\s*(\d+(?:\.\d+)?)\s*MHz', nvidia_smi_output, re.IGNORECASE)
                if graphics_match:
                    clocks['clock_graphics'] = int(float(graphics_match.group(1)))

                # SM clock fallback
                sm_match = re.search(r'SM\s*:\s*(\d+(?:\.\d+)?)\s*MHz', nvidia_smi_output, re.IGNORECASE)
                if sm_match:
                    clocks['clock_sm'] = int(float(sm_match.group(1)))

                # Memory clock fallback
                mem_match = re.search(r'Memory\s*:\s*(\d+(?:\.\d+)?)\s*MHz', nvidia_smi_output, re.IGNORECASE)
                if mem_match:
                    clocks['clock_memory'] = int(float(mem_match.group(1)))

                # Video clock fallback
                video_match = re.search(r'Video\s*:\s*(\d+(?:\.\d+)?)\s*MHz', nvidia_smi_output, re.IGNORECASE)
                if video_match:
                    clocks['clock_video'] = int(float(video_match.group(1)))

        except Exception as e:
            self.logger.log_error(e, {'context': 'extracting_clocks'})

        return clocks

    def _extract_pcie_info(self, nvidia_smi_output: str) -> Dict[str, int]:
        """Extract PCIe link information"""
        pcie_info = {
            'pcie_link_gen': 0,
            'pcie_link_width': 0
        }

        try:
            # Try multiple patterns for PCIe information - improved patterns
            patterns = [
                r'PCIe\s+Gen(\d+)\s+x(\d+)',
                r'PCIe\s+Generation\s+(\d+)\s*,\s*x(\d+)',
                r'PCIe\s+Gen\s*(\d+)\s*x\s*(\d+)',
                r'PCIe\s+Link\s+Gen\s*(\d+)\s+Width\s*x(\d+)',
                r'PCIe\s+Gen\s*(\d+)\s+/x(\d+)',  # Alternative format
                r'PCIe\s+(\d+)\s*x\s*(\d+)'  # Simple format
            ]

            for pattern in patterns:
                pcie_match = re.search(pattern, nvidia_smi_output, re.IGNORECASE)
                if pcie_match:
                    pcie_info['pcie_link_gen'] = int(pcie_match.group(1))
                    pcie_info['pcie_link_width'] = int(pcie_match.group(2))
                    break

            # If still not found, try to find PCIe info in different sections
            if pcie_info['pcie_link_gen'] == 0:
                # Look for PCIe information in any line
                lines = nvidia_smi_output.split('\n')
                for line in lines:
                    if 'pcie' in line.lower():
                        # Try to extract from any PCIe-related line
                        gen_match = re.search(r'gen\s*(\d+)', line, re.IGNORECASE)
                        width_match = re.search(r'x\s*(\d+)', line, re.IGNORECASE)
                        if gen_match and width_match:
                            pcie_info['pcie_link_gen'] = int(gen_match.group(1))
                            pcie_info['pcie_link_width'] = int(width_match.group(1))
                            break

            # Validate PCIe values - only log error if we found values but they're invalid
            if pcie_info['pcie_link_gen'] > 0 and (pcie_info['pcie_link_gen'] < 1 or pcie_info['pcie_link_gen'] > 6):
                self.logger.log_error(ValueError(f"Invalid PCIe generation: {pcie_info['pcie_link_gen']}"), {'context': 'pcie_validation'})
                pcie_info['pcie_link_gen'] = 0
            if pcie_info['pcie_link_width'] > 0 and (pcie_info['pcie_link_width'] < 1 or pcie_info['pcie_link_width'] > 16):
                self.logger.log_error(ValueError(f"Invalid PCIe width: {pcie_info['pcie_link_width']}"), {'context': 'pcie_validation'})
                pcie_info['pcie_link_width'] = 0

        except Exception as e:
            self.logger.log_error(e, {'context': 'extracting_pcie_info'})

        return pcie_info

    def _get_hostname(self) -> str:
        """Get the hostname"""
        import socket
        try:
            return socket.gethostname()
        except:
            return 'unknown'

    def validate_gpu_telemetry(self, raw_data: str) -> bool:
        """
        Validate that the GPU telemetry data has required fields

        Args:
            raw_data: Raw GPU telemetry data

        Returns:
            True if valid, False otherwise
        """
        if not raw_data or not raw_data.strip():
            self.logger.log_error(
                ValueError("Empty GPU telemetry data"),
                {'context': 'validating_gpu_telemetry'}
            )
            return False

        # Check for basic nvidia-smi indicators
        required_indicators = ['NVIDIA-SMI', 'Driver Version', 'CUDA Version']
        for indicator in required_indicators:
            if indicator not in raw_data:
                self.logger.log_error(
                    ValueError(f"Missing required indicator: {indicator}"),
                    {'context': 'validating_gpu_telemetry'}
                )
                return False

        return True

    def extract_gpu_metrics(self, event: GPUTelemetryEvent) -> Dict[str, Any]:
        """
        Extract key GPU metrics from a telemetry event

        Args:
            event: Parsed GPU telemetry event

        Returns:
            Dictionary containing key GPU metrics
        """
        return {
            'gpu_id': event.gpu_id,
            'gpu_name': event.gpu_name,
            'gpu_utilization_percent': event.gpu_utilization_percent,
            'memory_used_mb': event.memory_used_mb,
            'memory_total_mb': event.memory_total_mb,
            'memory_utilization_percent': event.memory_utilization_percent,
            'temperature_celsius': event.temperature_celsius,
            'power_draw_watts': event.power_draw_watts,
            'power_efficiency': (event.power_draw_watts / event.power_limit_watts) * 100 if event.power_limit_watts > 0 else 0,
            'fan_speed_percent': event.fan_speed_percent,
            'clock_graphics_mhz': event.clock_graphics_mhz,
            'pcie_link_gen': event.pcie_link_gen,
            'pcie_link_width': event.pcie_link_width,
            'process_count': len(event.processes),
            'total_process_memory_mb': sum(p.get('memory_mb', 0) for p in event.processes),
            'event_type': event.event_type,
            'hostname': event.hostname
        }

# Global parser instance
nvidia_telemetry_parser = NVIDIATelemetryParser()
