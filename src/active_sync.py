"""
Active Sync Service for JPMorgan Financial APIs - Real-time Payments → Revenue/Assets
Uses Redis pubsub + WebSocket-like SSE for active synchronization.
"""
import asyncio
import json
import os
import redis
from typing import Dict, Any, Callable

try:
    from .payments_service import payments_service
except Exception:
    from payments_service import payments_service  # type: ignore

try:
    from .database_fixed import db_manager
except Exception:
    from database_fixed import db_manager  # type: ignore

try:
    from .logger import telemetry_logger
except Exception:
    from logger import telemetry_logger  # type: ignore

class ActiveSyncService:
    def __init__(self):
        self.redis_client = redis.from_url('redis://localhost:6379') if 'REDIS_URL' in os.environ else None
        self.subscribers: Dict[str, list[Callable]] = {}
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    async def payment_completed_sync(self, payment_id: str):
        """Sync payment completion to revenue/assets real-time"""
        payment = payments_service.get_payment(payment_id)
        if not payment or payment.status != 'COMPLETED':
            return

        # Update revenue
        revenue_data = {
            'payment_id': payment_id,
            'amount': payment.amount,
            'user_id': payment.user_id,
            'timestamp': payment.updated_at.isoformat()
        }
        await db_manager.create_revenue(revenue_data)

        # Update assets (example: credit business asset)
        business_id = await self._get_user_business(payment.user_id)
        if business_id:
            await db_manager.update_business_asset_balance(business_id, payment.amount)

        # Broadcast via Redis pubsub
        if self.redis_client:
            self.redis_client.publish('payments:sync', json.dumps({
                'event': 'payment_synced',
                'payment_id': payment_id,
                'revenue_id': revenue_data['id'],
                'business_id': business_id
            }))

        telemetry_logger.get_logger().info(f"Synced payment {payment_id} to revenue/assets")

    async def _get_user_business(self, user_id: str) -> str:
        businesses = db_manager.get_businesses_by_user(user_id)
        return businesses[0].id if businesses else None

    def subscribe_sync_channel(self, channel: str, callback: Callable):
        """Subscribe to sync channel for real-time updates"""
        if self.redis_client:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(channel, callback=callback)
            if self.loop and self.loop.is_running():
                self.loop.create_task(asyncio.to_thread(pubsub.run_in_thread))
            else:
                pubsub.run_in_thread()

# Global active sync instance
active_sync = ActiveSyncService()

# Hook into payments_service (add to process_payment)
def hook_payment_sync(payment_id: str):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(active_sync.payment_completed_sync(payment_id))
    except RuntimeError:
        asyncio.run(active_sync.payment_completed_sync(payment_id))

