"""
OWLBAN GROUP AI API Server
FastAPI-based REST API for all AI services with NVIDIA GPU acceleration
"""

import logging
import os
from datetime import datetime, timezone
import secrets
import time
from typing import Annotated, Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# Import AI systems
try:
    from combined_nim_owlban_ai import CombinedSystem
    COMBINED_SYSTEM_AVAILABLE = True
except Exception:
    COMBINED_SYSTEM_AVAILABLE = False

try:
    from combined_nim_owlban_ai.ngc_catalog import NGCatalogManager
    NGCATALOG_AVAILABLE = True
except Exception:
    NGCATALOG_AVAILABLE = False

try:
    from combined_nim_owlban_ai.nim import NimManager
except Exception:
    NimManager = None

try:
    from new_products.revenue_optimizer import NVIDIARevenueOptimizer
    REVENUE_OPTIMIZER_AVAILABLE = True
except Exception:
    REVENUE_OPTIMIZER_AVAILABLE = False

try:
    from performance_optimization.reinforcement_learning_agent import ReinforcementLearningAgent
    RL_AGENT_AVAILABLE = True
except Exception:
    RL_AGENT_AVAILABLE = False

# Import database manager
try:
    from database_manager import DatabaseManager
    DB_MANAGER_AVAILABLE = True
except ImportError:
    DB_MANAGER_AVAILABLE = False

# Constants
REVENUE_OPTIMIZER_NOT_AVAILABLE = "Revenue optimizer not available"

# Security
security = HTTPBasic()
API_USERNAME = os.getenv("API_USERNAME", "owlban_admin")
API_PASSWORD = os.getenv("API_PASSWORD")
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

if API_PASSWORD is None:
    raise RuntimeError("API_PASSWORD environment variable must be set for API server authentication")

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify API credentials"""
    correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def _get_monitoring_stats() -> Dict[str, Any]:
    """Return shared monitoring statistics stored on the FastAPI app state."""
    if not hasattr(fastapi_app.state, "monitoring"):
        fastapi_app.state.monitoring = {"request_count": 0, "response_times": []}
    return fastapi_app.state.monitoring


# Monitoring middleware
class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request monitoring and logging."""

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        start_time = time.time()

        # Log request
        logger.info("Request: %s %s from %s", request.method, request.url.path, request.client.host)

        # Process request
        response = await call_next(request)

        # Calculate response time
        process_time = time.time() - start_time
        monitoring_stats = _get_monitoring_stats()
        monitoring_stats["request_count"] += 1
        monitoring_stats["response_times"].append(process_time)

        # Keep only last 100 response times
        if len(monitoring_stats["response_times"]) > 100:
            monitoring_stats["response_times"].pop(0)

        # Log response
        logger.info("Response: %s in %.3fs", response.status_code, process_time)

        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-API-Version"] = "1.0.0"

        return response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

fastapi_app = FastAPI(
    title="OWLBAN GROUP AI API",
    description="Unified API for NVIDIA-accelerated AI services",
    version="1.0.0"
)

fastapi_app.state.combined_system = None
fastapi_app.state.nim_manager = None
fastapi_app.state.revenue_optimizer = None
fastapi_app.state.rl_agent = None
fastapi_app.state.db_manager = None
fastapi_app.state.ngc_catalog_manager = None

# Add middleware
fastapi_app.add_middleware(MonitoringMiddleware)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
@fastapi_app.on_event("startup")
async def startup_event():
    """Initialize AI systems on application startup."""
    logger.info("Initializing AI systems...")

    if COMBINED_SYSTEM_AVAILABLE and CombinedSystem is not None:
        try:
            fastapi_app.state.combined_system = CombinedSystem()
            logger.info("CombinedSystem initialized")
        except Exception:
            logger.exception("Failed to initialize CombinedSystem")

    if REVENUE_OPTIMIZER_AVAILABLE and NimManager is not None and NVIDIARevenueOptimizer is not None:
        try:
            fastapi_app.state.nim_manager = NimManager()
            fastapi_app.state.nim_manager.initialize()
            fastapi_app.state.revenue_optimizer = NVIDIARevenueOptimizer(fastapi_app.state.nim_manager)
            logger.info("Revenue optimizer initialized")
        except Exception:
            logger.exception("Failed to initialize revenue optimizer")

    if RL_AGENT_AVAILABLE and ReinforcementLearningAgent is not None:
        try:
            fastapi_app.state.rl_agent = ReinforcementLearningAgent(['optimize', 'scale', 'monitor'])
            logger.info("RL agent initialized")
        except Exception:
            logger.exception("Failed to initialize RL agent")

    if DB_MANAGER_AVAILABLE:
        try:
            fastapi_app.state.db_manager = DatabaseManager()
            logger.info("Database manager initialized")
        except Exception:
            logger.exception("Failed to initialize database manager")

    if fastapi_app.state.combined_system is not None and hasattr(fastapi_app.state.combined_system, "ngc_catalog_manager"):
        try:
            fastapi_app.state.ngc_catalog_manager = fastapi_app.state.combined_system.ngc_catalog_manager
            fastapi_app.state.ngc_catalog_manager.initialize()
            logger.info("NGC catalog manager attached to combined system")
        except Exception:
            logger.exception("Failed to attach NGC catalog manager")
    elif NGCATALOG_AVAILABLE:
        try:
            fastapi_app.state.ngc_catalog_manager = NGCatalogManager()
            fastapi_app.state.ngc_catalog_manager.initialize()
            logger.info("Standalone NGC catalog manager initialized")
        except Exception:
            logger.exception("Failed to initialize standalone NGC catalog manager")

# Pydantic models
class RevenueOptimizationRequest(BaseModel):
    iterations: int = 10
    market_conditions: Optional[Dict[str, float]] = None

class InferenceRequest(BaseModel):
    data: Dict[str, Any]
    model_type: str = "prediction"

class SystemStatus(BaseModel):
    timestamp: str
    services: Dict[str, bool]
    gpu_status: Optional[Dict[str, Any]] = None
    database_status: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None

class LogEntry(BaseModel):
    level: str
    message: str
    timestamp: str
    source: str

# API endpoints
@fastapi_app.get("/")
async def root():
    return {"message": "OWLBAN GROUP AI API Server", "status": "running"}

@fastapi_app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@fastapi_app.get("/status", response_model=SystemStatus)
async def get_system_status():
    services = {
        "combined_system": fastapi_app.state.combined_system is not None,
        "revenue_optimizer": fastapi_app.state.revenue_optimizer is not None,
        "rl_agent": fastapi_app.state.rl_agent is not None,
        "nim_manager": fastapi_app.state.nim_manager is not None,
        "db_manager": fastapi_app.state.db_manager is not None
    }

    gpu_status = None
    if fastapi_app.state.nim_manager:
        gpu_status = fastapi_app.state.nim_manager.get_resource_status()

    database_status = None
    if fastapi_app.state.db_manager:
        database_status = fastapi_app.state.db_manager.get_database_status()

    uptime = time.time() - getattr(fastapi_app.state, 'start_time', time.time())
    monitoring_stats = _get_monitoring_stats()
    response_times = monitoring_stats.get("response_times", [])
    avg_response_time = sum(response_times) / max(len(response_times), 1)
    monitoring = {
        "uptime": uptime,
        "request_count": monitoring_stats.get("request_count", 0),
        "avg_response_time": avg_response_time
    }

    return SystemStatus(
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services,
        gpu_status=gpu_status,
        database_status=database_status,
        monitoring=monitoring
    )

@fastapi_app.get("/catalog/summary")
async def get_catalog_summary():
    manager = fastapi_app.state.ngc_catalog_manager
    if manager is None:
        raise HTTPException(status_code=503, detail="NGC catalog manager not available")

    return manager.get_catalog_summary() if hasattr(manager, "get_catalog_summary") else {"error": "catalog manager unavailable"}


@fastapi_app.get("/catalog/search")
async def search_catalog(query: str):
    manager = fastapi_app.state.ngc_catalog_manager
    if manager is None:
        raise HTTPException(status_code=503, detail="NGC catalog manager not available")

    return manager.search(query) if hasattr(manager, "search") else []


@fastapi_app.post("/revenue/optimize", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def optimize_revenue(request: RevenueOptimizationRequest, background_tasks: BackgroundTasks):
    if not fastapi_app.state.revenue_optimizer:
        raise HTTPException(status_code=503, detail=REVENUE_OPTIMIZER_NOT_AVAILABLE)

    try:
        background_tasks.add_task(fastapi_app.state.revenue_optimizer.optimize_revenue, request.iterations)
        return {
            "message": "Revenue optimization started with %d iterations" % request.iterations,
            "status": "running"
        }
    except Exception:
        logger.exception("Revenue optimization failed")
        raise HTTPException(status_code=500, detail="Revenue optimization failed")

@fastapi_app.get("/revenue/profit", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def get_current_profit():
    if not fastapi_app.state.revenue_optimizer:
        raise HTTPException(status_code=503, detail=REVENUE_OPTIMIZER_NOT_AVAILABLE)

    try:
        profit = fastapi_app.state.revenue_optimizer.get_current_profit()
        return {"current_profit": profit}
    except Exception:
        logger.exception("Failed to get profit")
        raise HTTPException(status_code=500, detail="Failed to get profit")

@fastapi_app.post("/inference", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def run_inference(request: InferenceRequest):
    if not fastapi_app.state.combined_system:
        raise HTTPException(status_code=503, detail="Combined system not available")

    try:
        result = fastapi_app.state.combined_system.run_inference(request.data)
        return {"result": result}
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed")

@fastapi_app.post("/rl/learn", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def rl_learn(state: List[float], action: str, reward: float, next_state: List[float]):
    if not fastapi_app.state.rl_agent:
        raise HTTPException(status_code=503, detail="RL agent not available")

    try:
        fastapi_app.state.rl_agent.learn(state, action, reward, next_state)
        return {"message": "RL learning completed"}
    except Exception:
        logger.exception("RL learning failed")
        raise HTTPException(status_code=500, detail="RL learning failed")

@fastapi_app.post("/rl/action", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def get_rl_action(state: List[float]):
    if not fastapi_app.state.rl_agent:
        raise HTTPException(status_code=503, detail="RL agent not available")

    try:
        action = fastapi_app.state.rl_agent.choose_action(state)
        return {"action": action}
    except Exception:
        logger.exception("RL action selection failed")
        raise HTTPException(status_code=500, detail="RL action selection failed")

@fastapi_app.get("/gpu/status", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def get_gpu_status():
    if not fastapi_app.state.nim_manager:
        raise HTTPException(status_code=503, detail="NIM manager not available")

    try:
        gpu_status = fastapi_app.state.nim_manager.get_resource_status()
        return {"gpu_status": gpu_status}
    except Exception:
        logger.exception("GPU status check failed")
        raise HTTPException(status_code=500, detail="GPU status check failed")

@fastapi_app.get("/quantum/portfolio", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def get_quantum_portfolio():
    if not fastapi_app.state.revenue_optimizer:
        raise HTTPException(status_code=503, detail=REVENUE_OPTIMIZER_NOT_AVAILABLE)

    try:
        result = fastapi_app.state.revenue_optimizer.optimize_quantum_portfolio()
        return {"portfolio": result.__dict__}
    except Exception:
        logger.exception("Quantum portfolio optimization failed")
        raise HTTPException(status_code=500, detail="Quantum portfolio optimization failed")

@fastapi_app.get("/quantum/risk", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def get_quantum_risk():
    if not fastapi_app.state.revenue_optimizer:
        raise HTTPException(status_code=503, detail=REVENUE_OPTIMIZER_NOT_AVAILABLE)

    try:
        result = fastapi_app.state.revenue_optimizer.analyze_quantum_risk()
        return {"risk_analysis": result.__dict__}
    except Exception:
        logger.exception("Quantum risk analysis failed")
        raise HTTPException(status_code=500, detail="Quantum risk analysis failed")

@fastapi_app.get("/quantum/predict/{symbol}", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def predict_market(symbol: str):
    if not fastapi_app.state.revenue_optimizer:
        raise HTTPException(status_code=503, detail=REVENUE_OPTIMIZER_NOT_AVAILABLE)

    try:
        prediction = fastapi_app.state.revenue_optimizer.predict_market_with_quantum(symbol)
        return {"prediction": prediction.__dict__}
    except Exception:
        logger.exception("Quantum market prediction failed")
        raise HTTPException(status_code=500, detail="Quantum market prediction failed")

@fastapi_app.get("/logs")
def get_logs(username: Annotated[str, Depends(verify_credentials)], lines: int = 100):
    """Get recent log entries (admin only)"""
    logger.info("Logs requested by %s", username)

    try:
        with open('api_server.log', 'r', encoding='utf-8') as f:
            logs = f.readlines()[-lines:]
        return {"logs": logs, "user": username}
    except FileNotFoundError:
        return {"logs": ["No log file found"], "user": username}
    except Exception:
        logger.exception("Failed to read logs")
        raise HTTPException(status_code=500, detail="Failed to read logs")

@fastapi_app.post("/logs")
async def add_log_entry(entry: LogEntry, username: Annotated[str, Depends(verify_credentials)]):
    """Add a log entry"""
    logger.info("Log entry added by %s: %s", username, entry.message)

    log_level = getattr(logging, entry.level.upper(), logging.INFO)
    logger.log(log_level, "[%s] %s", entry.source, entry.message)

    return {"message": "Log entry added", "user": username}

@fastapi_app.get("/metrics", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def get_metrics(username: Annotated[str, Depends(verify_credentials)]):
    """Get system metrics"""
    logger.info("Metrics requested by %s", username)

    if not fastapi_app.state.db_manager:
        raise HTTPException(status_code=503, detail="Database manager not available")

    try:
        metrics = fastapi_app.state.db_manager.get_predictions(limit=50)
        return {"metrics": metrics, "user": username}
    except Exception:
        logger.exception("Failed to get metrics")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@fastapi_app.post("/metrics", responses={503: {"description": "Service unavailable"}, 500: {"description": "Internal server error"}})
async def save_metric(metric_name: str, value: float, username: Annotated[str, Depends(verify_credentials)], tags: Optional[Dict] = None):
    """Save a system metric"""
    logger.info("Metric saved by %s: %s = %f", username, metric_name, value)

    if not fastapi_app.state.db_manager:
        raise HTTPException(status_code=503, detail="Database manager not available")

    try:
        fastapi_app.state.db_manager.save_system_metric(metric_name, value, tags)
        return {"message": "Metric saved", "user": username}
    except Exception:
        logger.exception("Failed to save metric")
        raise HTTPException(status_code=500, detail="Failed to save metric")

if __name__ == "__main__":
    fastapi_app.state.start_time = time.time()
    logger.info("Starting OWLBAN GROUP AI API Server")
    uvicorn.run(fastapi_app, host=API_HOST, port=API_PORT)
