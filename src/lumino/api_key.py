"""
API key management endpoints for the Lumino SDK.

This module contains the ApiKeyEndpoint class, which provides methods
for interacting with API key-related endpoints.
"""

import logging
from typing import Any, List

from lumino.models import (
    ApiKeyCreate,
    ApiKeyUpdate,
    ApiKeyResponse,
    ApiKeyWithSecretResponse,
    ListResponse,
    Pagination
)


class ApiKeyEndpoint:
    """
    Handles API key-related endpoints for the Lumino SDK.
    """

    def __init__(self, sdk: Any):
        """
        Initialize the ApiKeyEndpoint.

        Args:
            sdk (Any): The LuminoSDK instance.
        """
        self._sdk = sdk
        self.logger = logging.getLogger(__name__)

    async def create_api_key(self, api_key_create: ApiKeyCreate) -> ApiKeyWithSecretResponse:
        """
        Create a new API key.

        Args:
            api_key_create (ApiKeyCreate): The API key creation data.

        Returns:
            ApiKeyWithSecretResponse: The created API key information, including the secret.

        Raises:
            LuminoAPIError: If the API request fails.
            LuminoValidationError: If the provided data is invalid.
        """
        self.logger.info("Creating new API key: %s", api_key_create.name)
        data = await self._sdk._request("POST", "/api-keys", json=api_key_create.dict())
        return ApiKeyWithSecretResponse(**data)

    async def list_api_keys(self, page: int = 1, items_per_page: int = 20) -> ListResponse:
        """
        List all API keys.

        Args:
            page (int): The page number to retrieve. Defaults to 1.
            items_per_page (int): The number of items per page. Defaults to 20.

        Returns:
            ListResponse: A list of API keys and pagination information.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Listing API keys (page %d)", page)
        params = {"page": page, "items_per_page": items_per_page}
        data = await self._sdk._request("GET", "/api-keys", params=params)
        return ListResponse(
            data=[ApiKeyResponse(**item) for item in data['data']],
            pagination=Pagination(**data['pagination'])
        )

    async def get_api_key(self, key_name: str) -> ApiKeyResponse:
        """
        Get information about a specific API key.

        Args:
            key_name (str): The name of the API key.

        Returns:
            ApiKeyResponse: The API key information.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Getting API key: %s", key_name)
        data = await self._sdk._request("GET", f"/api-keys/{key_name}")
        return ApiKeyResponse(**data)

    async def update_api_key(self, key_name: str, api_key_update: ApiKeyUpdate) -> ApiKeyResponse:
        """
        Update an API key.

        Args:
            key_name (str): The name of the API key to update.
            api_key_update (ApiKeyUpdate): The updated API key information.

        Returns:
            ApiKeyResponse: The updated API key information.

        Raises:
            LuminoAPIError: If the API request fails.
            LuminoValidationError: If the provided data is invalid.
        """
        self.logger.info("Updating API key: %s", key_name)
        data = await self._sdk._request(
            "PATCH",
            f"/api-keys/{key_name}",
            json=api_key_update.dict(exclude_unset=True)
        )
        return ApiKeyResponse(**data)

    async def revoke_api_key(self, key_name: str) -> ApiKeyResponse:
        """
        Revoke an API key.

        Args:
            key_name (str): The name of the API key to revoke.

        Returns:
            ApiKeyResponse: The revoked API key information.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Revoking API key: %s", key_name)
        data = await self._sdk._request("DELETE", f"/api-keys/{key_name}")
        return ApiKeyResponse(**data)