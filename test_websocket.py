#!/usr/bin/env python3
"""
WebSocket Testing Script for JPMorgan Financial APIs
Tests WebSocket connection and live data synchronization
"""
import json
import time
import threading
from flask import Flask
from flask_socketio import SocketIO, emit
import socketio

def test_websocket_connection():
    """Test WebSocket connection and events"""
    print("Testing WebSocket connection...")

    # Create SocketIO client
    sio = socketio.Client()

    # Event handlers
    @sio.event
    def connect():
        print("✅ Connected to WebSocket server")

    @sio.event
    def disconnect():
        print("❌ Disconnected from WebSocket server")

    @sio.event
    def connection_established(data):
        print(f"📡 Connection established: {data}")

    @sio.event
    def response(data):
        print(f"📨 Response received: {data}")

    @sio.event
    def live_data_update(data):
        print(f"🔄 Live data update: {data}")

    @sio.event
    def sync_complete(data):
        print(f"✅ Sync complete: {data}")

    @sio.event
    def realtime_metrics(data):
        print(f"📊 Real-time metrics: {data}")

    @sio.event
    def sync_error(data):
        print(f"❌ Sync error: {data}")

    @sio.event
    def metrics_error(data):
        print(f"❌ Metrics error: {data}")

    try:
        # Connect to server
        sio.connect('http://localhost:5000')
        time.sleep(1)  # Wait for connection

        # Test basic message
        print("Testing test_message event...")
        sio.emit('test_message', {'message': 'Hello from test script!', 'timestamp': time.time()})
        time.sleep(1)

        # Test live data sync
        print("Testing sync_live_data event...")
        sio.emit('sync_live_data', {'sync_type': 'full', 'timestamp': time.time()})
        time.sleep(2)

        # Test real-time metrics
        print("Testing get_realtime_metrics event...")
        sio.emit('get_realtime_metrics')
        time.sleep(2)

        # Wait a bit more for responses
        time.sleep(2)

        # Disconnect
        sio.disconnect()
        print("✅ WebSocket test completed successfully")

    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

    return True

def test_websocket_status_endpoint():
    """Test the /ws/status endpoint"""
    print("Testing /ws/status endpoint...")
    try:
        import requests
        response = requests.get('http://localhost:5000/ws/status')
        if response.status_code == 200:
            data = response.json()
            print(f"✅ WebSocket status: {data}")
            return True
        else:
            print(f"❌ WebSocket status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ WebSocket status test failed: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Starting WebSocket Testing Suite")
    print("=" * 50)

    # Test status endpoint first
    status_ok = test_websocket_status_endpoint()

    if status_ok:
        # Test WebSocket connection
        ws_ok = test_websocket_connection()

        if ws_ok:
            print("\n🎉 All WebSocket tests passed!")
        else:
            print("\n❌ WebSocket connection tests failed!")
    else:
        print("\n❌ WebSocket status endpoint test failed!")

    print("=" * 50)
    print("WebSocket testing completed.")
