"""
WebSocket Manager for Real-time Data Streaming
"""
import asyncio
import json
import logging
from typing import Dict, Set, Any
from datetime import datetime, timezone
import websockets
from websockets.exceptions import ConnectionClosed
import redis
import uuid
import os

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time telemetry data streaming"""

    def __init__(self, redis_url: str = None):
        redis_url = os.getenv('REDIS_URL') or 'redis://localhost:6379'
        self.active_connections: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()
        except Exception as e:
            logger.warning(f"Redis not available at {redis_url}: {str(e)}. WebSocket manager will work without Redis.")
            self.redis_client = None
            self.pubsub = None

    async def register_connection(self, websocket: websockets.WebSocketServerProtocol, client_id: str):
        """Register a new WebSocket connection"""
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()

        self.active_connections[client_id].add(websocket)
        logger.info(f"WebSocket connection registered for client: {client_id}")

    async def unregister_connection(self, websocket: websockets.WebSocketServerProtocol, client_id: str):
        """Unregister a WebSocket connection"""
        if client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]

        logger.info(f"WebSocket connection unregistered for client: {client_id}")

    async def broadcast_to_client(self, client_id: str, message: Dict[str, Any]):
        """Broadcast message to specific client"""
        if client_id in self.active_connections:
            message['timestamp'] = datetime.now(timezone.utc).isoformat()
            message_json = json.dumps(message)

            disconnected = set()
            for websocket in self.active_connections[client_id]:
                try:
                    await websocket.send(message_json)
                except ConnectionClosed:
                    disconnected.add(websocket)

            # Clean up disconnected websockets
            for websocket in disconnected:
                await self.unregister_connection(websocket, client_id)

    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        message['timestamp'] = datetime.now(timezone.utc).isoformat()
        message_json = json.dumps(message)

        disconnected_clients = set()
        for client_id, websockets_set in self.active_connections.items():
            disconnected = set()
            for websocket in websockets_set:
                try:
                    await websocket.send(message_json)
                except ConnectionClosed:
                    disconnected.add(websocket)

            # Clean up disconnected websockets
            for websocket in disconnected:
                websockets_set.discard(websocket)

            if not websockets_set:
                disconnected_clients.add(client_id)

        # Clean up empty client entries
        for client_id in disconnected_clients:
            del self.active_connections[client_id]

    async def publish_telemetry_event(self, telemetry_data: Dict[str, Any]):
        """Publish telemetry event to Redis channel for broadcasting"""
        if self.redis_client is None:
            logger.warning("Redis not available, skipping telemetry event publish")
            return
        try:
            # Add metadata
            event = {
                'type': 'telemetry_event',
                'data': telemetry_data,
                'event_id': str(uuid.uuid4()),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            # Publish to Redis channel
            self.redis_client.publish('telemetry_stream', json.dumps(event))

            logger.info(f"Published telemetry event to stream: {event['event_id']}")

        except Exception as e:
            logger.error(f"Error publishing telemetry event: {str(e)}")

    async def publish_anomaly_alert(self, anomaly_data: Dict[str, Any]):
        """Publish anomaly alert to Redis channel"""
        if self.redis_client is None:
            logger.warning("Redis not available, skipping anomaly alert publish")
            return
        try:
            alert = {
                'type': 'anomaly_alert',
                'data': anomaly_data,
                'alert_id': str(uuid.uuid4()),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'severity': anomaly_data.get('severity', 'medium')
            }

            # Publish to Redis channel
            self.redis_client.publish('telemetry_stream', json.dumps(alert))

            logger.info(f"Published anomaly alert to stream: {alert['alert_id']}")

        except Exception as e:
            logger.error(f"Error publishing anomaly alert: {str(e)}")

    async def publish_system_status(self, status_data: Dict[str, Any]):
        """Publish system status update"""
        if self.redis_client is None:
            logger.warning("Redis not available, skipping system status publish")
            return
        try:
            status = {
                'type': 'system_status',
                'data': status_data,
                'status_id': str(uuid.uuid4()),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            # Publish to Redis channel
            self.redis_client.publish('telemetry_stream', json.dumps(status))

            logger.info(f"Published system status to stream: {status['status_id']}")

        except Exception as e:
            logger.error(f"Error publishing system status: {str(e)}")

    async def handle_redis_messages(self):
        """Handle messages from Redis pub/sub"""
        if self.pubsub is None:
            logger.warning("Redis not available, skipping Redis message handler")
            return
        try:
            self.pubsub.subscribe('telemetry_stream')
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self.broadcast_to_all(data)
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON received from Redis stream")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {str(e)}")

        except Exception as e:
            logger.error(f"Error in Redis message handler: {str(e)}")

    async def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())

    async def get_client_count(self) -> int:
        """Get number of unique clients"""
        return len(self.active_connections)

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.pubsub is not None:
                self.pubsub.close()
            if self.redis_client is not None:
                self.redis_client.close()
            logger.info("WebSocket manager cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

async def websocket_handler(websocket, path):
    """Handle WebSocket connections"""
    client_id = str(uuid.uuid4())
    logger.info(f"New WebSocket connection: {client_id}")

    try:
        await websocket_manager.register_connection(websocket, client_id)

        # Send welcome message
        welcome_message = {
            'type': 'connection_established',
            'client_id': client_id,
            'message': 'Connected to telemetry stream'
        }
        await websocket.send(json.dumps(welcome_message))

        # Keep connection alive and handle messages
        async for message in websocket:
            try:
                data = json.loads(message)
                # Handle client messages if needed
                logger.debug(f"Received message from client {client_id}: {data}")
            except json.JSONDecodeError:
                error_message = {
                    'type': 'error',
                    'message': 'Invalid JSON format'
                }
                await websocket.send(json.dumps(error_message))

    except ConnectionClosed:
        logger.info(f"WebSocket connection closed: {client_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler for {client_id}: {str(e)}")
    finally:
        await websocket_manager.unregister_connection(websocket, client_id)

async def start_websocket_server_async(host: str = "localhost", port: int = 8765):
    """Start the WebSocket server asynchronously"""
    try:
        # Start the WebSocket server
        server = await websockets.serve(websocket_handler, host, port)
        logger.info(f"WebSocket server started on ws://{host}:{port}")

        # Start Redis message handler concurrently
        redis_task = asyncio.create_task(websocket_manager.handle_redis_messages())

        # Wait for server to close
        await server.wait_closed()

        # Cancel Redis task
        redis_task.cancel()
        try:
            await redis_task
        except asyncio.CancelledError:
            pass

    except KeyboardInterrupt:
        logger.info("WebSocket server shutting down...")
    except Exception as e:
        logger.error(f"Error starting WebSocket server: {str(e)}")
    finally:
        websocket_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(start_websocket_server_async())
