"""
Simple logger for the application
"""
import logging
import traceback


class Logger:
    def __init__(self):
        self.logger = logging.getLogger('telemetry_logger')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, msg):
        self.logger.info(msg)

    def log_info(self, msg, context=None):
        """
        Log an info message with optional context
        
        Args:
            msg: The message to log
            context: Optional dictionary with additional context
        """
        if context:
            msg = f"{msg} | Context: {context}"
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def log_error(self, exception, context=None):
        """
        Log an error with exception details and optional context

        Args:
            exception: The exception object
            context: Optional dictionary with additional context
        """
        error_msg = f"Error: {str(exception)}"
        if context:
            error_msg += f" | Context: {context}"
        self.logger.error(error_msg)
        self.logger.debug(traceback.format_exc())

    def debug(self, msg):
        """Log debug message"""
        self.logger.debug(msg)

    def log_debug(self, msg, context=None):
        """Log debug with context"""
        if context:
            msg = f"{msg} | Context: {context}"
        self.logger.debug(msg)

    def get_logger(self):
        return self.logger

    def warning(self, msg):
        """Log warning message"""
        self.logger.warning(msg)



telemetry_logger = Logger()
