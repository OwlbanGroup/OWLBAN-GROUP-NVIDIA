"""
NVIDIA Triton Inference Server Integration
Unified model serving for all AI products with GPU acceleration
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any
import json
import time

# Optional Triton client imports with fallbacks
try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
    TRITON_AVAILABLE = True
except ImportError:
    httpclient = None
    grpcclient = None
    TRITON_AVAILABLE = False
    logging.warning("Triton client not available, using fallback mode")

class TritonInferenceServer:
    """NVIDIA Triton Inference Server for unified model serving"""

    def __init__(self, server_url: str = "localhost:8000", use_grpc: bool = True):
        self.server_url = server_url
        self.use_grpc = use_grpc
        self.logger = logging.getLogger("TritonInferenceServer")

        # Initialize clients with fallback
        if TRITON_AVAILABLE:
            if use_grpc:
                self.client = grpcclient.InferenceServerClient(url=server_url)
            else:
                self.client = httpclient.InferenceServerClient(url=server_url)
        else:
            self.client = None
            self.logger.warning("Triton client not available, using mock mode")

        self.models = {}
        self.logger.info("Triton Inference Server initialized at %s", server_url)

    def load_model(self, model_name: str, model_version: str = "1") -> bool:
        """Load model into Triton server"""
        try:
            # Check if model is available
            if not self.client.is_model_ready(model_name, model_version):
                self.logger.error("Model %s:%s not ready", model_name, model_version)
                return False

            self.models[model_name] = {
                'version': model_version,
                'metadata': self.client.get_model_metadata(model_name, model_version),
                'config': self.client.get_model_config(model_name, model_version)
            }

            self.logger.info("Model %s:%s loaded successfully", model_name, model_version)
            return True

        except Exception as e:
            self.logger.error("Failed to load model %s: %s", model_name, e)
            return False

    def infer(self, model_name: str, inputs: Dict[str, np.ndarray],
              outputs: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Perform inference using Triton server"""
        try:
            # Prepare inputs
            triton_inputs = []
            for input_name, input_data in inputs.items():
                if self.use_grpc:
                    triton_input = grpcclient.InferInput(input_name, input_data.shape, "FP32")
                    triton_input.set_data_from_numpy(input_data)
                else:
                    triton_input = httpclient.InferInput(input_name, input_data.shape, "FP32")
                    triton_input.set_data_from_numpy(input_data)
                triton_inputs.append(triton_input)

            # Prepare outputs
            triton_outputs = []
            if outputs is None:
                # Use all outputs from model metadata
                model_meta = self.models.get(model_name, {}).get('metadata', {})
                output_names = [output['name'] for output in model_meta.get('outputs', [])]
            else:
                output_names = outputs

            for output_name in output_names:
                if self.use_grpc:
                    triton_output = grpcclient.InferRequestedOutput(output_name)
                else:
                    triton_output = httpclient.InferRequestedOutput(output_name)
                triton_outputs.append(triton_output)

            # Perform inference
            model_version = self.models.get(model_name, {}).get('version', "1")
            results = self.client.infer(model_name, triton_inputs, outputs=triton_outputs, model_version=model_version)

            # Extract results
            output_data = {}
            for output_name in output_names:
                output_data[output_name] = results.as_numpy(output_name)

            return output_data

        except Exception as e:
            self.logger.error("Inference failed for model %s: %s", model_name, e)
            return {}

    def unload_model(self, model_name: str) -> bool:
        """Unload model from Triton server"""
        try:
            # Note: Triton doesn't have direct unload, but we can remove from our tracking
            if model_name in self.models:
                del self.models[model_name]
                self.logger.info("Model %s unloaded from tracking", model_name)
                return True
            return False
        except Exception as e:
            self.logger.error("Failed to unload model %s: %s", model_name, e)
            return False

    def get_server_status(self) -> Dict[str, Any]:
        """Get Triton server status"""
        try:
            server_live = self.client.is_server_live()
            server_ready = self.client.is_server_ready()
            server_metadata = self.client.get_server_metadata()

            return {
                'server_live': server_live,
                'server_ready': server_ready,
                'server_metadata': server_metadata,
                'loaded_models': list(self.models.keys())
            }
        except Exception as e:
            self.logger.error("Failed to get server status: %s", e)
            return {'error': str(e)}

class TritonModelManager:
    """Manager for Triton-based model serving across all AI products"""

    def __init__(self, triton_server: TritonInferenceServer):
        self.triton = triton_server
        self.logger = logging.getLogger("TritonModelManager")
        self.model_registry = {}

    def register_model(self, product_name: str, model_name: str,
                      input_specs: Dict[str, tuple], output_specs: Dict[str, tuple]):
        """Register model for a specific AI product"""
        self.model_registry[product_name] = {
            'model_name': model_name,
            'input_specs': input_specs,
            'output_specs': output_specs,
            'loaded': False
        }
        self.logger.info("Registered model %s for %s", model_name, product_name)

    def load_product_models(self, product_name: str) -> bool:
        """Load all models for a specific AI product"""
        if product_name not in self.model_registry:
            self.logger.error("Product %s not registered", product_name)
            return False

        model_name = self.model_registry[product_name]['model_name']
        if self.triton.load_model(model_name):
            self.model_registry[product_name]['loaded'] = True
            self.logger.info("Loaded models for %s", product_name)
            return True
        return False

    def infer_product(self, product_name: str, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Perform inference for a specific AI product"""
        if product_name not in self.model_registry or not self.model_registry[product_name]['loaded']:
            self.logger.error("Product %s models not loaded", product_name)
            return {}

        model_name = self.model_registry[product_name]['model_name']
        return self.triton.infer(model_name, inputs)

    def get_product_status(self, product_name: str) -> Dict[str, Any]:
        """Get status of models for a specific product"""
        if product_name not in self.model_registry:
            return {'error': f'Product {product_name} not registered'}

        registry = self.model_registry[product_name]
        return {
            'product_name': product_name,
            'model_name': registry['model_name'],
            'loaded': registry['loaded'],
            'input_specs': registry['input_specs'],
            'output_specs': registry['output_specs']
        }
