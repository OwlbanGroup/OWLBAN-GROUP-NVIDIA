"""
OWLBAN GROUP AI API Server
FastAPI-based REST API for all AI services with NVIDIA GPU acceleration
"""

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime
import secrets
import time

# Import AI systems
try:
    from combined_nim_owlban_ai import CombinedSystem
    combined_system_available = True
except ImportError:
    combined_system_available = False

try:
    from new_products.revenue_optimizer import NVIDIARevenueOptimizer
    from combined_nim_owlban_ai.nim import NimManager
    revenue_optimizer_available = True
except ImportError:
    revenue_optimizer_available = False

try:
    from performance_optimization.reinforcement_learning_agent import ReinforcementLearningAgent
    rl_agent_available = True
except ImportError:
    rl_agent_available = False

# Import database manager
try:
    from database_manager import DatabaseManager
    db_manager_available = True
except ImportError:
    db_manager_available = False

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
    """Middleware for request monitoring and logging"""

    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.response_times = []

    async def dispatch(self, request, call_next):
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path} from {request.client.host}")

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
        logger.info(f"Response: {response.status_code} in {process_time:.3f}s")

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

app = FastAPI(
    title="OWLBAN GROUP AI API",
    description="Unified API for NVIDIA-accelerated AI services",
    version="1.0.0"
)

# Add middleware
app.add_middleware(MonitoringMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
combined_system = None
nim_manager = None
revenue_optimizer = None
rl_agent = None

# Initialize systems
@app.on_event("startup")
async def startup_event():
    global combined_system, nim_manager, revenue_optimizer, rl_agent

    logger.info("Initializing AI systems...")

    if combined_system_available:
        try:
            combined_system = CombinedSystem()
            logger.info("CombinedSystem initialized")
        except Exception as e:
            logger.error(f"Failed to initialize CombinedSystem: {e}")

    if revenue_optimizer_available:
        try:
            nim_manager = NimManager()
            nim_manager.initialize()
            revenue_optimizer = NVIDIARevenueOptimizer(nim_manager)
            logger.info("Revenue optimizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize revenue optimizer: {e}")

    if rl_agent_available:
        try:
            rl_agent = ReinforcementLearningAgent(['optimize', 'scale', 'monitor'])
            logger.info("RL agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RL agent: {e}")

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
@app.get("/")
async def root():
    return {"message": "OWLBAN GROUP AI API Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    services = {
        "combined_system": combined_system is not None,
        "revenue_optimizer": revenue_optimizer is not None,
        "rl_agent": rl_agent is not None,
        "nim_manager": nim_manager is not None
    }

    gpu_status = None
    if nim_manager:
        gpu_status = nim_manager.get_resource_status()

    database_status = None
    if db_manager:
        database_status = db_manager.get_database_status()

    monitoring = {
        "uptime": time.time() - getattr(app.state, 'start_time', time.time()),
        "request_count": getattr(app.middleware_stack.app.user_middleware[0].app, 'request_count', 0),
        "avg_response_time": sum(getattr(app.middleware_stack.app.user_middleware[0].app, 'response_times', [])) / max(len(getattr(app.middleware_stack.app.user_middleware[0].app, 'response_times', [])), 1)
    }

    return SystemStatus(
        timestamp=datetime.utcnow().isoformat(),
        services=services,
        gpu_status=gpu_status,
        database_status=database_status,
        monitoring=monitoring
    )

@app.post("/revenue/optimize")
async def optimize_revenue(request: RevenueOptimizationRequest, background_tasks: BackgroundTasks):
    if not revenue_optimizer:
        raise HTTPException(status_code=503, detail="Revenue optimizer not available")

    try:
        # Run optimization in background
        background_tasks.add_task(revenue_optimizer.optimize_revenue, request.iterations)

        return {
            "message": f"Revenue optimization started with {request.iterations} iterations",
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Revenue optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/revenue/profit")
async def get_current_profit():
    if not revenue_optimizer:
        raise HTTPException(status_code=503, detail="Revenue optimizer not available")

    try:
        profit = revenue_optimizer.get_current_profit()
        return {"current_profit": profit}
    except Exception as e:
        logger.error(f"Failed to get profit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference")
async def run_inference(request: InferenceRequest):
    if not combined_system:
        raise HTTPException(status_code=503, detail="Combined system not available")

    try:
        result = combined_system.run_inference(request.data)
        return {"result": result}
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rl/learn")
async def rl_learn(state: List[float], action: str, reward: float, next_state: List[float]):
    if not rl_agent:
        raise HTTPException(status_code=503, detail="RL agent not available")

    try:
        rl_agent.learn(state, action, reward, next_state)
        return {"message": "RL learning completed"}
    except Exception as e:
        logger.error(f"RL learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rl/action")
async def get_rl_action(state: List[float]):
    if not rl_agent:
        raise HTTPException(status_code=503, detail="RL agent not available")

    try:
        action = rl_agent.choose_action(state)
        return {"action": action}
    except Exception as e:
        logger.error(f"RL action selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gpu/status")
async def get_gpu_status():
    if not nim_manager:
        raise HTTPException(status_code=503, detail="NIM manager not available")

    try:
        status = nim_manager.get_resource_status()
        return {"gpu_status": status}
    except Exception as e:
        logger.error(f"GPU status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quantum/portfolio")
async def get_quantum_portfolio():
    if not revenue_optimizer:
        raise HTTPException(status_code=503, detail="Revenue optimizer not available")

    try:
        result = revenue_optimizer.optimize_quantum_portfolio()
        return {"portfolio": result.__dict__}
    except Exception as e:
        logger.error(f"Quantum portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quantum/risk")
async def get_quantum_risk():
    if not revenue_optimizer:
        raise HTTPException(status_code=503, detail="Revenue optimizer not available")

    try:
        result = revenue_optimizer.analyze_quantum_risk()
        return {"risk_analysis": result.__dict__}
    except Exception as e:
        logger.error(f"Quantum risk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quantum/predict/{symbol}")
async def predict_market(symbol: str):
    if not revenue_optimizer:
        raise HTTPException(status_code=503, detail="Revenue optimizer not available")

    try:
        prediction = revenue_optimizer.predict_market_with_quantum(symbol)
        return {"prediction": prediction.__dict__}
    except Exception as e:
        logger.error(f"Quantum market prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
async def get_logs(lines: int = 100, username: str = Depends(verify_credentials)):
    """Get recent log entries (admin only)"""
    logger.info(f"Logs requested by {username}")

    try:
        with open('api_server.log', 'r') as f:
            logs = f.readlines()[-lines:]
        return {"logs": logs, "user": username}
    except FileNotFoundError:
        return {"logs": ["No log file found"], "user": username}
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/logs")
async def add_log_entry(entry: LogEntry, username: str = Depends(verify_credentials)):
    """Add a log entry"""
    logger.info(f"Log entry added by {username}: {entry.message}")

    # Log the entry
    log_level = getattr(logging, entry.level.upper(), logging.INFO)
    logger.log(log_level, f"[{entry.source}] {entry.message}")

    return {"message": "Log entry added", "user": username}

@app.get("/metrics")
async def get_metrics(username: str = Depends(verify_credentials)):
    """Get system metrics"""
    logger.info(f"Metrics requested by {username}")

    if not db_manager:
        raise HTTPException(status_code=503, detail="Database manager not available")

    try:
        # Get recent system metrics
        metrics = db_manager.get_predictions(limit=50)  # Using predictions as proxy for metrics
        return {"metrics": metrics, "user": username}
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics")
async def save_metric(metric_name: str, value: float, tags: Optional[Dict] = None, username: str = Depends(verify_credentials)):
    """Save a system metric"""
    logger.info(f"Metric saved by {username}: {metric_name} = {value}")

    if not db_manager:
        raise HTTPException(status_code=503, detail="Database manager not available")

    try:
        db_manager.save_system_metric(metric_name, value, tags)
        return {"message": "Metric saved", "user": username}
    except Exception as e:
        logger.error(f"Failed to save metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Set start time for uptime tracking
    app.state.start_time = time.time()

    logger.info("Starting OWLBAN GROUP AI API Server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
