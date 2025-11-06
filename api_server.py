"""
OWLBAN GROUP AI API Server
FastAPI-based REST API for all AI services with NVIDIA GPU acceleration
"""

import logging
from datetime import datetime, timezone
import secrets
import time
from typing import Dict, List, Optional, Any

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
except ImportError:
    COMBINED_SYSTEM_AVAILABLE = False

try:
    from new_products.revenue_optimizer import NVIDIARevenueOptimizer
    from combined_nim_owlban_ai.nim import NimManager
    REVENUE_OPTIMIZER_AVAILABLE = True
except ImportError:
    REVENUE_OPTIMIZER_AVAILABLE = False

try:
    from performance_optimization.reinforcement_learning_agent import ReinforcementLearningAgent
    RL_AGENT_AVAILABLE = True
except ImportError:
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
API_USERNAME = "owlban_admin"
API_PASSWORD = "quantum_secure_2024"

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

# Monitoring middleware
class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request monitoring and logging."""

    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.response_times = []

    async def dispatch(self, request, call_next):
        start_time = time.time()

        # Log request
        logger.info("Request: %s %s from %s", request.method, request.url.path, request.client.host)

        # Process request
        response = await call_next(request)

        # Calculate response time
        process_time = time.time() - start_time
        self.request_count += 1
        self.response_times.append(process_time)

        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times.pop(0)

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

# Add middleware
fastapi_app.add_middleware(MonitoringMiddleware)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
COMBINED_SYSTEM: Optional[Any] = None
NIM_MANAGER: Optional[Any] = None
REVENUE_OPTIMIZER: Optional[Any] = None
RL_AGENT: Optional[Any] = None
DB_MANAGER: Optional[Any] = None

# Initialize systems
@fastapi_app.on_event("startup")
async def startup_event():
    """Initialize AI systems on application startup."""
    global COMBINED_SYSTEM, NIM_MANAGER, REVENUE_OPTIMIZER, RL_AGENT, DB_MANAGER

    logger.info("Initializing AI systems...")

    if COMBINED_SYSTEM_AVAILABLE:
        try:
            COMBINED_SYSTEM = CombinedSystem()
            logger.info("CombinedSystem initialized")
        except Exception as e:
            logger.error("Failed to initialize CombinedSystem: %s", e)

    if REVENUE_OPTIMIZER_AVAILABLE:
        try:
            NIM_MANAGER = NimManager()
            NIM_MANAGER.initialize()
            REVENUE_OPTIMIZER = NVIDIARevenueOptimizer(NIM_MANAGER)
            logger.info("Revenue optimizer initialized")
        except Exception as e:
            logger.error("Failed to initialize revenue optimizer: %s", e)

    if RL_AGENT_AVAILABLE:
        try:
            RL_AGENT = ReinforcementLearningAgent(['optimize', 'scale', 'monitor'])
            logger.info("RL agent initialized")
        except Exception as e:
            logger.error("Failed to initialize RL agent: %s", e)

    if DB_MANAGER_AVAILABLE:
        try:
            DB_MANAGER = DatabaseManager()
            logger.info("Database manager initialized")
        except Exception as e:
            logger.error("Failed to initialize database manager: %s", e)

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
        "combined_system": COMBINED_SYSTEM is not None,
        "revenue_optimizer": REVENUE_OPTIMIZER is not None,
        "rl_agent": RL_AGENT is not None,
        "nim_manager": NIM_MANAGER is not None,
        "db_manager": DB_MANAGER is not None
    }

    gpu_status = None
    if NIM_MANAGER:
        gpu_status = NIM_MANAGER.get_resource_status()

    database_status = None
    if DB_MANAGER:
        database_status = DB_MANAGER.get_database_status()

    uptime = time.time() - getattr(fastapi_app.state, 'start_time', time.time())
    request_count = getattr(fastapi_app.middleware_stack.app.user_middleware[0].app, 'request_count', 0)
    response_times = getattr(fastapi_app.middleware_stack.app.user_middleware[0].app, 'response_times', [])
    avg_response_time = sum(response_times) / max(len(response_times), 1)
    monitoring = {
        "uptime": uptime,
        "request_count": request_count,
        "avg_response_time": avg_response_time
    }

    return SystemStatus(
        timestamp=datetime.now(timezone.utc).isoformat(),
        services=services,
        gpu_status=gpu_status,
        database_status=database_status,
        monitoring=monitoring
    )

@fastapi_app.post("/revenue/optimize")
async def optimize_revenue(request: RevenueOptimizationRequest, background_tasks: BackgroundTasks):
    if not REVENUE_OPTIMIZER:
        raise HTTPException(status_code=503, detail=REVENUE_OPTIMIZER_NOT_AVAILABLE)

    try:
        # Run optimization in background
        background_tasks.add_task(REVENUE_OPTIMIZER.optimize_revenue, request.iterations)

        return {
            "message": "Revenue optimization started with %d iterations" % request.iterations,
            "status": "running"
        }
    except Exception as e:
        logger.error("Revenue optimization failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.get("/revenue/profit")
async def get_current_profit():
    if not REVENUE_OPTIMIZER:
        raise HTTPException(status_code=503, detail=REVENUE_OPTIMIZER_NOT_AVAILABLE)

    try:
        profit = REVENUE_OPTIMIZER.get_current_profit()
        return {"current_profit": profit}
    except Exception as e:
        logger.error("Failed to get profit: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.post("/inference")
async def run_inference(request: InferenceRequest):
    if not COMBINED_SYSTEM:
        raise HTTPException(status_code=503, detail="Combined system not available")

    try:
        result = COMBINED_SYSTEM.run_inference(request.data)
        return {"result": result}
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.post("/rl/learn")
async def rl_learn(state: List[float], action: str, reward: float, next_state: List[float]):
    if not RL_AGENT:
        raise HTTPException(status_code=503, detail="RL agent not available")

    try:
        RL_AGENT.learn(state, action, reward, next_state)
        return {"message": "RL learning completed"}
    except Exception as e:
        logger.error("RL learning failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.post("/rl/action")
async def get_rl_action(state: List[float]):
    if not RL_AGENT:
        raise HTTPException(status_code=503, detail="RL agent not available")

    try:
        action = RL_AGENT.choose_action(state)
        return {"action": action}
    except Exception as e:
        logger.error("RL action selection failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.get("/gpu/status")
async def get_gpu_status():
    if not NIM_MANAGER:
        raise HTTPException(status_code=503, detail="NIM manager not available")

    try:
        gpu_status = NIM_MANAGER.get_resource_status()
        return {"gpu_status": gpu_status}
    except Exception as e:
        logger.error("GPU status check failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.get("/quantum/portfolio")
async def get_quantum_portfolio():
    if not REVENUE_OPTIMIZER:
        raise HTTPException(status_code=503, detail=REVENUE_OPTIMIZER_NOT_AVAILABLE)

    try:
        result = REVENUE_OPTIMIZER.optimize_quantum_portfolio()
        return {"portfolio": result.__dict__}
    except Exception as e:
        logger.error("Quantum portfolio optimization failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.get("/quantum/risk")
async def get_quantum_risk():
    if not REVENUE_OPTIMIZER:
        raise HTTPException(status_code=503, detail=REVENUE_OPTIMIZER_NOT_AVAILABLE)

    try:
        result = REVENUE_OPTIMIZER.analyze_quantum_risk()
        return {"risk_analysis": result.__dict__}
    except Exception as e:
        logger.error("Quantum risk analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.get("/quantum/predict/{symbol}")
async def predict_market(symbol: str):
    if not REVENUE_OPTIMIZER:
        raise HTTPException(status_code=503, detail=REVENUE_OPTIMIZER_NOT_AVAILABLE)

    try:
        prediction = REVENUE_OPTIMIZER.predict_market_with_quantum(symbol)
        return {"prediction": prediction.__dict__}
    except Exception as e:
        logger.error("Quantum market prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.get("/logs")
def get_logs(lines: int = 100, username: str = Depends(verify_credentials)):
    """Get recent log entries (admin only)"""
    logger.info("Logs requested by %s", username)

    try:
        with open('api_server.log', 'r', encoding='utf-8') as f:
            logs = f.readlines()[-lines:]
        return {"logs": logs, "user": username}
    except FileNotFoundError:
        return {"logs": ["No log file found"], "user": username}
    except Exception as e:
        logger.error("Failed to read logs: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.post("/logs")
async def add_log_entry(entry: LogEntry, username: str = Depends(verify_credentials)):
    """Add a log entry"""
    logger.info("Log entry added by %s: %s", username, entry.message)

    # Log the entry
    log_level = getattr(logging, entry.level.upper(), logging.INFO)
    logger.log(log_level, "[%s] %s", entry.source, entry.message)

    return {"message": "Log entry added", "user": username}

@fastapi_app.get("/metrics")
async def get_metrics(username: str = Depends(verify_credentials)):
    """Get system metrics"""
    logger.info("Metrics requested by %s", username)

    if not DB_MANAGER:
        raise HTTPException(status_code=503, detail="Database manager not available")

    try:
        # Get recent system metrics
        metrics = DB_MANAGER.get_predictions(limit=50)  # Using predictions as proxy for metrics
        return {"metrics": metrics, "user": username}
    except Exception as e:
        logger.error("Failed to get metrics: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

@fastapi_app.post("/metrics")
async def save_metric(metric_name: str, value: float, tags: Optional[Dict] = None, username: str = Depends(verify_credentials)):
    """Save a system metric"""
    logger.info("Metric saved by %s: %s = %f", username, metric_name, value)

    if not DB_MANAGER:
        raise HTTPException(status_code=503, detail="Database manager not available")

    try:
        DB_MANAGER.save_system_metric(metric_name, value, tags)
        return {"message": "Metric saved", "user": username}
    except Exception as e:
        logger.error("Failed to save metric: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e

if __name__ == "__main__":
    # Set start time for uptime tracking
    fastapi_app.state.start_time = time.time()

    logger.info("Starting OWLBAN GROUP AI API Server")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
