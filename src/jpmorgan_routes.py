"""
JP Morgan API Integration Routes
Exposes JP Morgan Payments APIs through our system
"""
from datetime import datetime
from typing import Optional, Dict, Any

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Depends, Query

from .jpmorgan_client import get_jpmorgan_client, JPMorganAPIClient
from shared.schemas import APIResponse
from shared.auth import require_auth, TokenData

logger = structlog.get_logger()

router = APIRouter(prefix="/api/jpmorgan", tags=["JP Morgan Integration"])


# AI ACCOUNTS Routes
@router.get("/accounts")
async def get_accounts(
    account_type: str = Query(
        "all",
        description="Account type: corporate, business, personal, or all"
    ),
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Get accounts from JP Morgan AI ACCOUNTS project"""
    try:
        accounts = await client.get_accounts(account_type)
        return APIResponse(
            status="success",
            message=f"Retrieved {len(accounts)} accounts",
            data={"accounts": accounts}
        )
    except httpx.HTTPError as e:
        logger.error("Failed to get accounts", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve accounts"
        ) from e


@router.get("/accounts/{account_id}/balance")
async def get_account_balance(
    account_id: str,
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Get account balance from JP Morgan"""
    try:
        balance = await client.get_account_balance(account_id)
        return APIResponse(
            status="success",
            message="Balance retrieved successfully",
            data=balance
        )
    except httpx.HTTPError as e:
        logger.error("Failed to get balance", account_id=account_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve balance"
        ) from e


@router.get("/accounts/{account_id}/transactions")
async def get_account_transactions(
    account_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000),
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Get account transactions from JP Morgan"""
    try:
        transactions = await client.get_account_transactions(
            account_id, start_date, end_date, limit
        )
        return APIResponse(
            status="success",
            message=f"Retrieved {len(transactions)} transactions",
            data={"transactions": transactions}
        )
    except httpx.HTTPError as e:
        logger.error(
            "Failed to get transactions",
            account_id=account_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve transactions"
        ) from e


# CORPORATE LOGIN Routes
@router.post("/corporate/login")
async def corporate_login(
    credentials: Dict[str, str],
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Corporate executive login through JP Morgan"""
    try:
        username = credentials.get("username")
        password = credentials.get("password")

        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password required")

        result = await client.corporate_login(username, password)
        return APIResponse(
            status="success",
            message="Login successful",
            data=result
        )
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        logger.error("Corporate login failed", error=str(e))
        raise HTTPException(status_code=401, detail="Login failed") from e


@router.get("/corporate/users/{user_id}")
async def get_corporate_user(
    user_id: str,
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Get corporate user information from JP Morgan"""
    try:
        user_info = await client.get_corporate_user_info(user_id)
        return APIResponse(
            status="success",
            message="User information retrieved",
            data=user_info
        )
    except httpx.HTTPError as e:
        logger.error("Failed to get user info", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve user information"
        ) from e


# PAYROLL Routes
@router.get("/payroll")
async def get_payroll(
    employee_id: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Get payroll data from JP Morgan OWL PAYROLL project"""
    try:
        payroll_data = await client.get_payroll_data(employee_id, start_date, end_date)
        return APIResponse(
            status="success",
            message=f"Retrieved {len(payroll_data)} payroll records",
            data={"payroll": payroll_data}
        )
    except httpx.HTTPError as e:
        logger.error("Failed to get payroll data", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve payroll data"
        ) from e


@router.post("/payroll/process")
async def process_payroll(
    payroll_data: Dict[str, Any],
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Process payroll payment through JP Morgan"""
    try:
        result = await client.process_payroll(payroll_data)
        return APIResponse(
            status="success",
            message="Payroll processed successfully",
            data=result
        )
    except httpx.HTTPError as e:
        logger.error("Failed to process payroll", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to process payroll"
        ) from e


# PETTY CASH Routes
@router.get("/petty-cash/balance")
async def get_petty_cash_balance(
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Get petty cash balance from JP Morgan OWL PETTY CASH project"""
    try:
        balance = await client.get_petty_cash_balance()
        return APIResponse(
            status="success",
            message="Petty cash balance retrieved",
            data=balance
        )
    except httpx.HTTPError as e:
        logger.error("Failed to get petty cash balance", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve petty cash balance"
        ) from e


@router.post("/petty-cash/requests")
async def create_petty_cash_request(
    request_data: Dict[str, Any],
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Create petty cash request through JP Morgan"""
    try:
        result = await client.create_petty_cash_request(request_data)
        return APIResponse(
            status="success",
            message="Petty cash request created",
            data=result
        )
    except httpx.HTTPError as e:
        logger.error("Failed to create petty cash request", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to create petty cash request"
        ) from e


@router.get("/petty-cash/transactions")
async def get_petty_cash_transactions(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Get petty cash transactions from JP Morgan"""
    try:
        transactions = await client.get_petty_cash_transactions(start_date, end_date)
        return APIResponse(
            status="success",
            message=f"Retrieved {len(transactions)} transactions",
            data={"transactions": transactions}
        )
    except httpx.HTTPError as e:
        logger.error("Failed to get petty cash transactions", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve transactions"
        ) from e


# OWL1 DATA INTEGRATION Routes
@router.post("/integration/sync/{data_type}")
async def sync_data(
    data_type: str,
    data: Dict[str, Any],
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Sync data with JP Morgan Owl1 integration"""
    try:
        result = await client.sync_data(data_type, data)
        return APIResponse(
            status="success",
            message=f"Data synced successfully: {data_type}",
            data=result
        )
    except httpx.HTTPError as e:
        logger.error("Failed to sync data", data_type=data_type, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to sync data") from e


@router.get("/integration/status")
async def get_integration_status(
    _token_data: TokenData = Depends(require_auth),
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Get Owl1 integration status from JP Morgan"""
    try:
        status = await client.get_integration_status()
        return APIResponse(
            status="success",
            message="Integration status retrieved",
            data=status
        )
    except httpx.HTTPError as e:
        logger.error("Failed to get integration status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve integration status"
        ) from e


# Health check for JP Morgan integration
@router.get("/health")
async def jpmorgan_health(
    client: JPMorganAPIClient = Depends(get_jpmorgan_client)
):
    """Check JP Morgan API integration health"""
    try:
        # Test connection to each project
        projects_status = {}

        for project_name in client.projects.keys():
            try:
                token = await client.get_access_token(project_name)
                projects_status[project_name] = {
                    "status": "connected",
                    "has_token": bool(token)
                }
            except httpx.HTTPError as proj_error:
                projects_status[project_name] = {
                    "status": "error",
                    "error": str(proj_error)
                }


        all_connected = all(
            p["status"] == "connected"
            for p in projects_status.values()
        )


        return APIResponse(
            status="success" if all_connected else "partial",
            message="JP Morgan integration health check",
            data={
                "overall_status": "healthy" if all_connected else "degraded",
                "projects": projects_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    except httpx.HTTPError as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail="Health check failed") from e
