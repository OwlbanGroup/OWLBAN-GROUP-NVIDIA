#!/usr/bin/env python
"""Check NVIDIA GPU status for OWLBAN-GROUP-NVIDIA project"""
import pynvml
import sys

pynvml.nvmlInit()

# Get GPU count
count = pynvml.nvmlDeviceGetCount()
print(f"GPUs available: {count}")

for i in range(count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    
    # Get GPU name
    name = pynvml.nvmlDeviceGetName(handle)
    print(f"\nGPU {i}: {name}")
    
    # Get memory info
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"  Memory: {info.total // 1024**3} GB total")
    
    # Get UUID
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    print(f"  UUID: {uuid}")
    
    # Get GPU mode status
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"  GPU Utilization: {util.gpu}%")
    print(f"  Memory Utilization: {util.memory}%")
    
    # Get temperature
    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        print(f"  Temperature: {temp} C")
    except:
        print("  Temperature: N/A")

pynvml.nvmlShutdown()
print("\nGPU check complete.")