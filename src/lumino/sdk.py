"""
Core module for the Lumino SDK.

This module contains the main LuminoSDK class and common utilities used across the SDK.
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime

import aiohttp

from lumino.billing import BillingEndpoint
from lumino.exceptions import LuminoAPIError
from lumino.user import UserEndpoint
from lumino.api_key import ApiKeyEndpoint
from lumino.dataset import DatasetEndpoint
from lumino.fine_tuning import FineTuningEndpoint
from lumino.model import ModelEndpoint
from lumino.usage import UsageEndpoint

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""

    def default(self, obj: Any) -> str:
        """
        Encode datetime objects as ISO format strings.

        Args:
            obj (Any): The object to encode.

        Returns:
            str: The encoded object.
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class LuminoSDK:
    """
    Main class for interacting with the Lumino API.

    This class provides methods to interact with various endpoints of the Lumino API,
    including user management, API key management, dataset operations, fine-tuning jobs,
    model information retrieval, and usage tracking.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.luminolabs.ai/v1"):
        """
        Initialize the LuminoSDK.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL of the Lumino API. Defaults to "https://api.luminolabs.ai/v1".
        """
        self._api_key = api_key  # Changed from self.api_key to self._api_key
        self.base_url = base_url
        self.session: aiohttp.ClientSession | None = None
        self.logger = logging.getLogger(__name__)

        # Initialize endpoint-specific classes
        self.user = UserEndpoint(self)
        self.api_keys = ApiKeyEndpoint(self)  # Changed from api_key to api_keys
        self.dataset = DatasetEndpoint(self)
        self.fine_tuning = FineTuningEndpoint(self)
        self.model = ModelEndpoint(self)
        self.usage = UsageEndpoint(self)
        self.billing = BillingEndpoint(self)

    async def __aenter__(self) -> 'LuminoSDK':
        """Set up the aiohttp session when entering an async context."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the aiohttp session when exiting an async context."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _ensure_session(self) -> None:
        """Ensure that an aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers={"X-API-Key": self._api_key})

    async def _request(self, method: str, endpoint: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Make an HTTP request to the Lumino API.

        Args:
            method (str): The HTTP method to use (e.g., "GET", "POST").
            endpoint (str): The API endpoint to call.
            **kwargs: Additional keyword arguments to pass to the request.

        Returns:
            Dict[str, Any]: The JSON response from the API.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"
        self.logger.debug("Making %s request to %s", method, url)

        if 'json' in kwargs:
            kwargs['data'] = json.dumps(kwargs.pop('json'), cls=DateTimeEncoder)
            kwargs['headers'] = kwargs.get('headers', {})
            kwargs['headers']['Content-Type'] = 'application/json'

        try:
            async with self.session.request(method, url, **kwargs) as response:  # type: ignore
                if response.status >= 400:
                    await self._handle_error_response(response)
                return await response.json()
        except aiohttp.ClientResponseError as e:
            raise LuminoAPIError(e.status, str(e))

    async def _handle_error_response(self, response: aiohttp.ClientResponse) -> None:
        """
        Handle error responses from the API.

        Args:
            response (aiohttp.ClientResponse): The response object from the API.

        Raises:
            LuminoAPIError: With detailed error information.
        """
        try:
            error_data = await response.json()
        except json.JSONDecodeError:
            error_data = await response.text()

        if isinstance(error_data, dict) and 'message' in error_data:
            message = error_data['message']
            details = error_data.get('details')
        else:
            message = str(error_data)
            details = None

        raise LuminoAPIError(response.status, message, details)
