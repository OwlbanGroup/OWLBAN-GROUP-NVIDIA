"""Quick validation helpers for immediate use"""
import re
from typing import Tuple

class QuickValidators:
    """Quick validation helpers"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email format"""
        if not email or '@' not in email:
            return False, "Invalid email format"
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return False, "Invalid email format"
        return True, "Valid"
    
    @staticmethod
    def validate_phone(phone: str) -> Tuple[bool, str]:
        """Validate phone number"""
        if not phone:
            return False, "Phone number required"
        # Remove common formatting
        clean_phone = re.sub(r'[^\d+]', '', phone)
        if len(clean_phone) < 10:
            return False, "Phone number too short"
        return True, "Valid"
    
    @staticmethod
    def validate_string_length(text: str, min_len: int = 1, max_len: int = 255) -> Tuple[bool, str]:
        """Validate string length"""
        if not text:
            return False, f"Text required (min {min_len} characters)"
        if len(text) < min_len:
            return False, f"Text too short (min {min_len} characters)"
        if len(text) > max_len:
            return False, f"Text too long (max {max_len} characters)"
        return True, "Valid"
    
    @staticmethod
    def validate_numeric_range(value: float, min_val: float = 0, max_val: float = None) -> Tuple[bool, str]:
        """Validate numeric range"""
        if value < min_val:
            return False, f"Value must be at least {min_val}"
        if max_val and value > max_val:
            return False, f"Value must be at most {max_val}"
        return True, "Valid"
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input"""
        if not text:
            return ""
        # Remove dangerous characters
        dangerous = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')']
        for char in dangerous:
            text = text.replace(char, '')
        return text.strip()
