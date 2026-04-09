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

    def audit_log_transaction(self, user_id, action, endpoint, ip_address, user_agent, request_data=None, response_status=None, details=None):
        """
        Log audit trail entry to database
        
        Args:
            user_id: User identifier
            action: Action performed (e.g. 'CREATE_BUSINESS', 'PAYMENT_CONFIRM')
            endpoint: API endpoint used
            ip_address: Client IP
            user_agent: User agent string
            request_data: Request payload (JSON serializable)
            response_status: HTTP status code
            details: Additional context
        """
        try:
            import psycopg2
            from psycopg2.extras import Json  # For JSONB
            
            # Use same conn string as app (from env or default)
            conn_str = "postgresql://jpmorgan:secure_password_123@localhost:5432/jpmorgan_api"  # Default local
            
            conn = psycopg2.connect(conn_str)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO audit_logs (user_id, action, endpoint, ip_address, user_agent, 
                                          request_data, response_status, details)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id or 'anonymous',
                    action,
                    endpoint,
                    ip_address,
                    user_agent[:1000],  # Truncate
                    Json(request_data) if request_data else None,
                    response_status,
                    Json(details) if details else None
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Audit log failed: {str(e)}")
        finally:
            if 'conn' in locals():
                conn.close()



telemetry_logger = Logger()
