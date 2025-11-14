"""
Simple logger for the application
"""
import logging

class Logger:
    def __init__(self):
        self.logger = logging.getLogger('telemetry_logger')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def get_logger(self):
        return self.logger

telemetry_logger = Logger()
