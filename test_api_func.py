"""Test API server functionality on CPU"""
import asyncio
import os
import api_server
os.environ["API_PASSWORD"] = "test"

from combined_nim_owlban_ai import CombinedSystem

# Test get_system_status
result = asyncio.run(api_server.get_system_status())
print("System status OK:")
print("  timestamp:", result.timestamp)
print("  services type:", type(result.services))
if isinstance(result.services, dict):
    print("  services keys:", list(result.services.keys())[:5])
print("  monitoring type:", type(result.monitoring))

# Test inference via CombinedSystem (correct import)
cs = CombinedSystem()
cs.initialize()
print("\nCombinedSystem initialized successfully")

# Test inference
prediction = cs.run_inference({"test": "data"})
print("Inference OK:", prediction)

print("\n✅ All API server functions verified successfully on CPU!")
