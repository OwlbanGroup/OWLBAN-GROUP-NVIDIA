"""
Quantum Financial Monitoring System
OWLBAN GROUP - Enterprise Real-time Monitoring and Alerting
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Slack and Teams Integration
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logging.warning("Slack SDK not available")

try:
    from O365 import Account, Message
    TEAMS_AVAILABLE = True
except ImportError:
    TEAMS_AVAILABLE = False
    logging.warning("Microsoft Teams SDK not available")

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

try:
    from combined_nim_owlban_ai.financial_data_manager import FinancialDataManager
    from combined_nim_owlban_ai.financial_data_config import FinancialDataConfig
    FINANCIAL_MANAGER_AVAILABLE = True
except Exception:
    FINANCIAL_MANAGER_AVAILABLE = False

@dataclass
class AlertConfig:
    severity: str
    threshold: float
    comparison: str
    message_template: str
    channels: List[str]
    cooldown: int = 300  # seconds

@dataclass
class MetricConfig:
    name: str
    description: str
    unit: str
    alert_configs: List[AlertConfig]

class QuantumMonitor:
    """Enterprise Quantum Financial Monitoring System"""

    def __init__(self, config: Dict):
        self.config = config
        self._setup_monitoring()
        self._setup_notifications()
        self._initialize_metrics()
        # Attempt to initialize FinancialDataManager if available
        self.financial_manager = None
        if FINANCIAL_MANAGER_AVAILABLE:
            try:
                # Prefer an explicit config path, otherwise use 'financial' subsection
                cfg_path = self.config.get('financial_config_path')
                if cfg_path:
                    fdconf = FinancialDataConfig(cfg_path)
                    # Build the minimal config dict expected by FinancialDataManager
                    fm_cfg = {
                        'refinitiv': getattr(fdconf, 'refinitiv').__dict__ if hasattr(fdconf, 'refinitiv') else {},
                        'bloomberg': getattr(fdconf, 'bloomberg').__dict__ if hasattr(fdconf, 'bloomberg') else {},
                        'market_data': getattr(fdconf, 'market_data').__dict__ if hasattr(fdconf, 'market_data') else {}
                    }
                    self.financial_manager = FinancialDataManager(fm_cfg)
                else:
                    fin_cfg = self.config.get('financial') or {}
                    if fin_cfg:
                        self.financial_manager = FinancialDataManager(fin_cfg)
            except Exception as e:
                logging.warning(f"Failed to initialize FinancialDataManager: {e}")
                self.financial_manager = None

    def _setup_monitoring(self):
        """Initialize Prometheus metrics server"""
        # Start Prometheus metrics server
        start_http_server(self.config.get('prometheus_port', 9090))
        
        # Core performance metrics
        self.quantum_circuit_latency = Histogram(
            'quantum_circuit_latency_seconds',
            'Latency of quantum circuit execution',
            ['circuit_type', 'hardware_provider']
        )
        
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id', 'operation_type']
        )
        
        self.quantum_error_rate = Gauge(
            'quantum_error_rate',
            'Quantum circuit error rate',
            ['circuit_type']
        )
        
        # Financial metrics
        self.portfolio_performance = Gauge(
            'portfolio_performance',
            'Portfolio performance metrics',
            ['metric_type']
        )
        
        self.trading_volume = Counter(
            'trading_volume_total',
            'Total trading volume',
            ['asset_type']
        )
        
        self.risk_metrics = Gauge(
            'risk_metrics',
            'Risk assessment metrics',
            ['risk_type']
        )

    def _setup_notifications(self):
        """Initialize notification channels"""
        if SLACK_AVAILABLE:
            self.slack = WebClient(token=self.config['slack_token'])
            
        if TEAMS_AVAILABLE:
            self.teams = Account(credentials=(
                self.config['teams_client_id'],
                self.config['teams_client_secret']
            ))
            
        self.last_alert = {}  # Track alert cooldowns

    def _initialize_metrics(self):
        """Initialize metric configurations"""
        self.metric_configs = {
            'quantum_circuit': MetricConfig(
                name='quantum_circuit',
                description='Quantum circuit performance',
                unit='seconds',
                alert_configs=[
                    AlertConfig(
                        severity='critical',
                        threshold=1.0,
                        comparison='gt',
                        message_template='Quantum circuit latency exceeds {threshold}s',
                        channels=['slack-critical', 'teams-ops']
                    )
                ]
            ),
            'gpu_utilization': MetricConfig(
                name='gpu_utilization',
                description='GPU utilization',
                unit='percent',
                alert_configs=[
                    AlertConfig(
                        severity='warning',
                        threshold=90.0,
                        comparison='gt',
                        message_template='GPU utilization above {threshold}%',
                        channels=['slack-warnings']
                    )
                ]
            ),
            # Add more metric configurations as needed
        }

    async def monitor_quantum_performance(self):
        """Monitor quantum circuit performance"""
        while True:
            try:
                # Collect quantum metrics
                circuit_metrics = await self._get_quantum_metrics()
                
                # Update Prometheus metrics
                for circuit_type, metrics in circuit_metrics.items():
                    self.quantum_circuit_latency.labels(
                        circuit_type=circuit_type,
                        hardware_provider=metrics['provider']
                    ).observe(metrics['latency'])
                    
                    self.quantum_error_rate.labels(
                        circuit_type=circuit_type
                    ).set(metrics['error_rate'])
                    
                # Check alerts
                await self._check_quantum_alerts(circuit_metrics)
                
            except Exception as e:
                logging.error(f"Error monitoring quantum performance: {e}")
            
            await asyncio.sleep(self.config.get('quantum_monitor_interval', 1))

    async def monitor_gpu_performance(self):
        """Monitor GPU performance metrics"""
        while True:
            try:
                # Collect GPU metrics
                gpu_metrics = await self._get_gpu_metrics()
                
                # Update Prometheus metrics
                for gpu_id, metrics in gpu_metrics.items():
                    self.gpu_utilization.labels(
                        gpu_id=gpu_id,
                        operation_type=metrics['operation']
                    ).set(metrics['utilization'])
                    
                # Check alerts
                await self._check_gpu_alerts(gpu_metrics)
                
            except Exception as e:
                logging.error(f"Error monitoring GPU performance: {e}")
            
            await asyncio.sleep(self.config.get('gpu_monitor_interval', 1))

    async def monitor_financial_metrics(self):
        """Monitor financial performance metrics"""
        while True:
            try:
                # Collect financial metrics
                financial_metrics = await self._get_financial_metrics()
                
                # Update Prometheus metrics
                self.portfolio_performance.labels(
                    metric_type='sharpe_ratio'
                ).set(financial_metrics['sharpe_ratio'])
                
                self.trading_volume.labels(
                    asset_type='total'
                ).inc(financial_metrics['trading_volume'])
                
                self.risk_metrics.labels(
                    risk_type='var'
                ).set(financial_metrics['var'])
                
                # Check alerts
                await self._check_financial_alerts(financial_metrics)
                
            except Exception as e:
                logging.error(f"Error monitoring financial metrics: {e}")
            
            await asyncio.sleep(self.config.get('financial_monitor_interval', 1))

    async def _send_alert(self, alert_config: AlertConfig, metric_value: float):
        """Send alert through configured channels"""
        current_time = time.time()
        alert_key = f"{alert_config.severity}_{alert_config.threshold}"
        
        # Check cooldown
        if alert_key in self.last_alert:
            if current_time - self.last_alert[alert_key] < alert_config.cooldown:
                return
        
        self.last_alert[alert_key] = current_time
        message = alert_config.message_template.format(
            threshold=alert_config.threshold,
            value=metric_value
        )
        
        for channel in alert_config.channels:
            if channel.startswith('slack-') and SLACK_AVAILABLE:
                try:
                    await self._send_slack_alert(channel, message, alert_config.severity)
                except Exception as e:
                    logging.error(f"Error sending Slack alert: {e}")
                    
            elif channel.startswith('teams-') and TEAMS_AVAILABLE:
                try:
                    await self._send_teams_alert(channel, message, alert_config.severity)
                except Exception as e:
                    logging.error(f"Error sending Teams alert: {e}")

    async def _send_slack_alert(self, channel: str, message: str, severity: str):
        """Send alert to Slack"""
        if SLACK_AVAILABLE:
            try:
                self.slack.chat_postMessage(
                    channel=channel.replace('slack-', '#'),
                    text=message,
                    attachments=[{
                        'color': self._get_severity_color(severity),
                        'fields': [
                            {
                                'title': 'Severity',
                                'value': severity.upper(),
                                'short': True
                            },
                            {
                                'title': 'Timestamp',
                                'value': datetime.now().isoformat(),
                                'short': True
                            }
                        ]
                    }]
                )
            except SlackApiError as e:
                logging.error(f"Error sending Slack alert: {e.response['error']}")

    async def _send_teams_alert(self, channel: str, message: str, severity: str):
        """Send alert to Microsoft Teams"""
        if TEAMS_AVAILABLE:
            try:
                # Create teams message card
                card = {
                    "type": "message",
                    "attachments": [{
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "type": "AdaptiveCard",
                            "body": [
                                {
                                    "type": "TextBlock",
                                    "text": f"**{severity.upper()} Alert**",
                                    "weight": "bolder",
                                    "size": "medium"
                                },
                                {
                                    "type": "TextBlock",
                                    "text": message,
                                    "wrap": True
                                }
                            ],
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "version": "1.2"
                        }
                    }]
                }
                
                # Send to teams channel
                self.teams.send_message(channel.replace('teams-', ''), card)
                
            except Exception as e:
                logging.error(f"Error sending Teams alert: {e}")

    @staticmethod
    def _get_severity_color(severity: str) -> str:
        """Get color code for severity level"""
        return {
            'critical': '#FF0000',
            'error': '#FFA500',
            'warning': '#FFFF00',
            'info': '#00FF00'
        }.get(severity.lower(), '#808080')

    async def _get_quantum_metrics(self) -> Dict:
        """Collect quantum circuit metrics"""
        # If there is a connected quantum backend we would pull metrics from it.
        # For now provide safe, realistic simulated metrics using configured circuit types.
        cfg = self.config.get('metrics', {}).get('quantum_circuit', {})
        circuit_types = cfg.get('circuit_types', ['optimization', 'prediction', 'risk-analysis'])
        provider = self.config.get('default_quantum_provider', 'ionq')
        metrics = {}
        for ct in circuit_types:
            # Simulate latency: base + small random jitter
            base_latency = 0.2 if ct == 'prediction' else (0.8 if ct == 'optimization' else 1.2)
            latency = float(max(0.001, np.random.normal(loc=base_latency, scale=0.1)))
            # Simulate error rate (quantum noise)
            error_rate = float(max(0.0, np.random.normal(loc=0.01, scale=0.005)))
            metrics[ct] = {
                'provider': provider,
                'latency': round(latency, 4),
                'error_rate': round(error_rate, 6)
            }
        return metrics

    async def _get_gpu_metrics(self) -> Dict:
        """Collect GPU performance metrics"""
        gpu_metrics = {}
        # Prefer NVML for real metrics
        if NVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_metrics[f'gpu{i}'] = {
                        'operation': 'quantum_processing',
                        'utilization': float(util.gpu),
                        'memory_used_mb': int(mem_info.used // 1024 // 1024),
                        'memory_total_mb': int(mem_info.total // 1024 // 1024),
                        'temperature': None
                    }
            except Exception as e:
                logging.debug(f"NVML metric collection failed: {e}")
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            # Best-effort using torch for memory stats; utilization not available
            try:
                for i in range(torch.cuda.device_count()):
                    mem_alloc = torch.cuda.memory_allocated(i)
                    mem_total = torch.cuda.get_device_properties(i).total_memory
                    utilization = float(np.random.uniform(30.0, 80.0))
                    gpu_metrics[f'gpu{i}'] = {
                        'operation': 'quantum_processing',
                        'utilization': round(utilization, 2),
                        'memory_used_mb': int(mem_alloc // 1024 // 1024),
                        'memory_total_mb': int(mem_total // 1024 // 1024),
                        'temperature': None
                    }
            except Exception as e:
                logging.debug(f"Torch-based GPU metric collection failed: {e}")
        else:
            # Fallback: simulated single-GPU metrics
            utilization = float(np.random.uniform(20.0, 70.0))
            mem_used = int(np.random.uniform(2000, 16000))
            mem_total = 24576
            gpu_metrics['gpu0'] = {
                'operation': 'quantum_processing',
                'utilization': round(utilization, 2),
                'memory_used_mb': mem_used,
                'memory_total_mb': mem_total,
                'temperature': None
            }

        return gpu_metrics

    async def _get_financial_metrics(self) -> Dict:
        """Collect financial performance metrics"""
        # Attempt to use FinancialDataManager for live metrics when available
        if getattr(self, 'financial_manager', None):
            try:
                symbols = self.config.get('financial', {}).get('symbols', ['SPY'])
                provider = self.config.get('financial', {}).get('provider', 'bloomberg')
                end = datetime.utcnow()
                start = end - timedelta(days=self.config.get('financial_lookback_days', 30))

                # Try to fetch historical data for the first symbol
                hist = None
                try:
                    hist = await self.financial_manager.get_historical_data(symbols, start, end, interval='1d', provider=provider)
                except Exception:
                    # Some FinancialDataManager implementations may not be async or may not be implemented
                    try:
                        # try sync call as a fallback
                        hist = self.financial_manager.get_historical_data(symbols, start, end, interval='1d', provider=provider)
                    except Exception:
                        hist = None

                if hist:
                    # Attempt to extract closing prices
                    closes = None
                    # hist may be a dict keyed by symbol -> list[dict] or similar
                    for s in symbols:
                        if s in hist and hist[s]:
                            data = hist[s]
                            if isinstance(data, list) and len(data) > 0:
                                first = data[0]
                                if isinstance(first, dict):
                                    # Try common keys
                                    key = None
                                    for k in ('close', 'Close', 'last', 'close_price'):
                                        if k in first:
                                            key = k
                                            break
                                    if key:
                                        closes = [float(r[key]) for r in data if key in r]
                                        break
                                else:
                                    # assume numeric list
                                    closes = [float(v) for v in data]
                                    break

                    if closes and len(closes) >= 2:
                        returns = np.diff(np.log(np.array(closes)))
                        sharpe = float(np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-9))
                        trading_volume = int(self.config.get('financial', {}).get('estimated_trading_volume', 0))
                        var = float(self.config.get('financial', {}).get('estimated_var', 0.0))
                        return {'sharpe_ratio': round(sharpe, 3), 'trading_volume': trading_volume, 'var': round(var, 6)}
            except Exception as e:
                logging.debug(f"Financial manager metrics failed: {e}")

        # Fallback: simulate metrics if no financial manager or retrieval failed
        sharpe = float(max(0.0, np.random.normal(loc=1.8, scale=0.3)))
        trading_volume = int(max(0, np.random.normal(loc=2000000, scale=500000)))
        var = float(max(0.0001, np.random.normal(loc=0.015, scale=0.005)))

        return {
            'sharpe_ratio': round(sharpe, 3),
            'trading_volume': trading_volume,
            'var': round(var, 6)
        }

    async def start_monitoring(self):
        """Start all monitoring tasks with automatic recovery"""
        monitoring_tasks = [
            self.monitor_quantum_performance(),
            self.monitor_gpu_performance(),
            self.monitor_financial_metrics()
        ]

        # Start monitoring with error recovery
        while True:
            try:
                await asyncio.gather(*monitoring_tasks)
            except Exception as e:
                self.logger.error("Monitoring tasks failed, restarting: %s", e)
                await asyncio.sleep(5)  # Wait before restart
                continue

    def __del__(self):
        """Cleanup monitoring resources"""
        # Cleanup tasks as needed
        pass