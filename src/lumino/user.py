"""
User management endpoints for the Lumino SDK.

This module contains the UserEndpoint class, which provides methods
for interacting with user-related API endpoints.
"""

import logging
from typing import Any, Dict

from lumino.models import UserUpdate, UserResponse


class UserEndpoint:
    """
    Handles user-related API endpoints for the Lumino SDK.
    """

    def __init__(self, sdk: Any):
        """
        Initialize the UserEndpoint.

        Args:
            sdk (Any): The LuminoSDK instance.
        """
        self._sdk = sdk
        self.logger = logging.getLogger(__name__)

    async def get_current_user(self) -> UserResponse:
        """
        Get the current user's information.

        Returns:
            UserResponse: The current user's information.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Getting current user information")
        data = await self._sdk._request("GET", "/users/me")
        return UserResponse(**data)

    async def update_current_user(self, user_update: UserUpdate) -> UserResponse:
        """
        Update the current user's information.

        Args:
            user_update (UserUpdate): The updated user information.

        Returns:
            UserResponse: The updated user information.

        Raises:
            LuminoAPIError: If the API request fails.
            LuminoValidationError: If the provided data is invalid.
        """
        self.logger.info("Updating current user information")
        data = await self._sdk._request(
            "PATCH",
            "/users/me",
            json=user_update.dict(exclude_unset=True)
        )
        return UserResponse(**data)

    async def delete_account(self) -> Dict[str, Any]:
        """
        Delete the current user's account.

        Returns:
            Dict[str, Any]: The API response confirming account deletion.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.warning("Deleting user account")
        return await self._sdk._request("DELETE", "/users/me")

    async def get_account_settings(self) -> Dict[str, Any]:
        """
        Get the current user's account settings.

        Returns:
            Dict[str, Any]: The user's account settings.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Getting user account settings")
        return await self._sdk._request("GET", "/users/me/settings")

    async def update_account_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the current user's account settings.

        Args:
            settings (Dict[str, Any]): The updated account settings.

        Returns:
            Dict[str, Any]: The updated account settings.

        Raises:
            LuminoAPIError: If the API request fails.
            LuminoValidationError: If the provided settings are invalid.
        """
        self.logger.info("Updating user account settings")
        return await self._sdk._request("PATCH", "/users/me/settings", json=settings)
