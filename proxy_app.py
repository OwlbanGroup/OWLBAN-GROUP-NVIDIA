from flask import Flask, jsonify, request
import requests
import websocket
import threading

app = Flask(__name__)
BASE_URL = "http://localhost:5000"  # Replace with actual endpoint if deployed

@app.route('/health-check', methods=['GET'])
def health_check():
    response = requests.get(f"{BASE_URL}/health")
    return jsonify(response.json())

@app.route('/send-telemetry', methods=['POST'])
def send_telemetry():
    payload = request.json
    response = requests.post(f"{BASE_URL}/telemetry", json=payload)
    return jsonify(response.json())

@app.route('/detect-anomalies', methods=['POST'])
def detect_anomalies():
    data = request.json
    response = requests.post(f"{BASE_URL}/ml/anomalies", json=data)
    return jsonify(response.json())

@app.route('/convert-data', methods=['POST'])
def convert_data():
    data = request.json
    response = requests.post(f"{BASE_URL}/data/convert", json=data)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(port=5001, debug=True)
