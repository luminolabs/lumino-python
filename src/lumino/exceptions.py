"""
Custom exception classes for the Lumino SDK.

This module defines exception classes specific to the Lumino SDK,
allowing for more precise error handling in client applications.
"""

from typing import Any, Optional


class LuminoError(Exception):
    """Base exception class for all Lumino SDK errors."""

    def __init__(self, message: str):
        """
        Initialize the LuminoError.

        Args:
            message (str): The error message.
        """
        self.message = message
        super().__init__(self.message)


class LuminoAPIError(LuminoError):
    """Exception raised for errors returned by the Lumino API."""

    def __init__(self, status: int, message: str, details: Optional[Any] = None):
        """
        Initialize the LuminoAPIError.

        Args:
            status (int): The HTTP status code of the error.
            message (str): The error message.
            details (Optional[Any]): Additional error details, if any.
        """
        self.status = status
        self.details = details
        super().__init__(f"Lumino API Error (Status {status}): {message}")


class LuminoConfigurationError(LuminoError):
    """Exception raised for configuration errors in the Lumino SDK."""

    def __init__(self, message: str):
        """
        Initialize the LuminoConfigurationError.

        Args:
            message (str): The error message describing the configuration issue.
        """
        super().__init__(f"Lumino SDK Configuration Error: {message}")


class LuminoValidationError(LuminoError):
    """Exception raised for validation errors in the Lumino SDK."""

    def __init__(self, message: str, field: Optional[str] = None):
        """
        Initialize the LuminoValidationError.

        Args:
            message (str): The error message describing the validation issue.
            field (Optional[str]): The name of the field that failed validation, if applicable.
        """
        self.field = field
        error_message = f"Validation Error: {message}"
        if field:
            error_message = f"{error_message} (Field: {field})"
        super().__init__(error_message)
