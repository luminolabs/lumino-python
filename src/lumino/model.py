"""
Model information retrieval endpoints for the Lumino SDK.

This module contains the ModelEndpoint class, which provides methods
for interacting with model-related API endpoints.
"""

import logging
from typing import Any, Optional

from lumino.models import (
    BaseModelResponse,
    FineTunedModelResponse,
    ListResponse,
    Pagination
)


class ModelEndpoint:
    """
    Handles model-related API endpoints for the Lumino SDK.
    """

    def __init__(self, sdk: Any):
        """
        Initialize the ModelEndpoint.

        Args:
            sdk (Any): The LuminoSDK instance.
        """
        self._sdk = sdk
        self.logger = logging.getLogger(__name__)

    async def list_base_models(self, page: int = 1, items_per_page: int = 20) -> ListResponse:
        """
        List all available base models.

        Args:
            page (int): The page number to retrieve. Defaults to 1.
            items_per_page (int): The number of items per page. Defaults to 20.

        Returns:
            ListResponse: A list of base models and pagination information.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Listing base models (page %d)", page)
        params = {"page": page, "items_per_page": items_per_page}
        data = await self._sdk._request("GET", "/models/base", params=params)
        return ListResponse(
            data=[BaseModelResponse(**item) for item in data['data']],
            pagination=Pagination(**data['pagination'])
        )

    async def get_base_model(self, model_name: str) -> BaseModelResponse:
        """
        Get information about a specific base model.

        Args:
            model_name (str): The name of the base model.

        Returns:
            BaseModelResponse: Information about the base model.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Getting base model: %s", model_name)
        data = await self._sdk._request("GET", f"/models/base/{model_name}")
        return BaseModelResponse(**data)

    async def list_fine_tuned_models(self, page: int = 1, items_per_page: int = 20) -> ListResponse:
        """
        List all fine-tuned models.

        Args:
            page (int): The page number to retrieve. Defaults to 1.
            items_per_page (int): The number of items per page. Defaults to 20.

        Returns:
            ListResponse: A list of fine-tuned models and pagination information.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Listing fine-tuned models (page %d)", page)
        params = {"page": page, "items_per_page": items_per_page}
        data = await self._sdk._request("GET", "/models/fine-tuned", params=params)
        return ListResponse(
            data=[FineTunedModelResponse(**item) for item in data['data']],
            pagination=Pagination(**data['pagination'])
        )

    async def get_fine_tuned_model(self, model_name: str) -> FineTunedModelResponse:
        """
        Get information about a specific fine-tuned model.

        Args:
            model_name (str): The name of the fine-tuned model.

        Returns:
            FineTunedModelResponse: Information about the fine-tuned model.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Getting fine-tuned model: %s", model_name)
        data = await self._sdk._request("GET", f"/models/fine-tuned/{model_name}")
        return FineTunedModelResponse(**data)

    async def delete_fine_tuned_model(self, model_name: str) -> None:
        """
        Delete a fine-tuned model.

        Args:
            model_name (str): The name of the fine-tuned model to delete.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Deleting fine-tuned model: %s", model_name)
        await self._sdk._request("DELETE", f"/models/fine-tuned/{model_name}")

    async def get_model_performance(self, model_name: str) -> dict:
        """
        Get performance metrics for a specific model.

        Args:
            model_name (str): The name of the model (base or fine-tuned).

        Returns:
            dict: A dictionary containing performance metrics for the model.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Getting performance metrics for model: %s", model_name)
        data = await self._sdk._request("GET", f"/models/{model_name}/performance")
        return data

    async def compare_models(self, model_names: list[str]) -> dict:
        """
        Compare performance metrics for multiple models.

        Args:
            model_names (list[str]): A list of model names to compare.

        Returns:
            dict: A dictionary containing comparative performance metrics for the specified models.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Comparing models: %s", ", ".join(model_names))
        data = await self._sdk._request("POST", "/models/compare", json={"models": model_names})
        return data
