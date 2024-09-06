import json
import logging
from typing import Dict, Any
from datetime import datetime

import aiohttp

from lumino.models import (
    UserUpdate, UserResponse, ApiKeyCreate, ApiKeyUpdate, ApiKeyResponse, ApiKeyWithSecretResponse,
    DatasetCreate, DatasetUpdate, DatasetResponse, FineTuningJobCreate, FineTuningJobResponse,
    FineTuningJobDetailResponse, BaseModelResponse, FineTunedModelResponse, UsageRecordResponse,
    TotalCostResponse, ListResponse, Pagination
)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class LuminoSDK:
    """
    SDK for interacting with the Lumino API.

    This class provides methods to interact with various endpoints of the Lumino API,
    including user management, API key management, dataset operations, fine-tuning jobs,
    model information retrieval, and usage tracking.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.lumino.ai/v1"):
        """
        Initialize the LuminoSDK.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL of the Lumino API. Defaults to "https://api.lumino.ai/v1".
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Set up the aiohttp session when entering an async context."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers={"X-API-Key": self.api_key})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the aiohttp session when exiting an async context."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _ensure_session(self):
        """Ensure that an aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers={"X-API-Key": self.api_key})

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request to the Lumino API.

        Args:
            method (str): The HTTP method to use (e.g., "GET", "POST").
            endpoint (str): The API endpoint to call.
            **kwargs: Additional keyword arguments to pass to the request.

        Returns:
            Dict[str, Any]: The JSON response from the API.

        Raises:
            aiohttp.ClientResponseError: If the API request fails.
        """
        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"
        self.logger.debug(f"Making {method} request to {url}")

        if 'json' in kwargs:
            kwargs['data'] = json.dumps(kwargs.pop('json'), cls=DateTimeEncoder)
            kwargs['headers'] = kwargs.get('headers', {})
            kwargs['headers']['Content-Type'] = 'application/json'

        async with self.session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    # User Management methods

    async def get_current_user(self) -> UserResponse:
        """
        Get the current user's information.

        Returns:
            UserResponse: The current user's information.
        """
        self.logger.info("Getting current user information")
        data = await self._request("GET", "/users/me")
        return UserResponse(**data)

    async def update_current_user(self, user_update: UserUpdate) -> UserResponse:
        """
        Update the current user's information.

        Args:
            user_update (UserUpdate): The updated user information.

        Returns:
            UserResponse: The updated user information.
        """
        self.logger.info("Updating current user information")
        data = await self._request("PATCH", "/users/me", json=user_update.dict(exclude_unset=True))
        return UserResponse(**data)

    # API Keys methods

    async def create_api_key(self, api_key_create: ApiKeyCreate) -> ApiKeyWithSecretResponse:
        """
        Create a new API key.

        Args:
            api_key_create (ApiKeyCreate): The API key creation data.

        Returns:
            ApiKeyWithSecretResponse: The created API key information, including the secret.
        """
        self.logger.info(f"Creating new API key: {api_key_create.name}")
        data = await self._request("POST", "/api-keys", json=api_key_create.dict())
        return ApiKeyWithSecretResponse(**data)

    async def list_api_keys(self, page: int = 1, items_per_page: int = 20) -> ListResponse:
        """
        List all API keys.

        Args:
            page (int): The page number to retrieve. Defaults to 1.
            items_per_page (int): The number of items per page. Defaults to 20.

        Returns:
            ListResponse: A list of API keys and pagination information.
        """
        self.logger.info(f"Listing API keys (page {page})")
        params = {"page": page, "items_per_page": items_per_page}
        data = await self._request("GET", "/api-keys", params=params)
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
        """
        self.logger.info(f"Getting API key: {key_name}")
        data = await self._request("GET", f"/api-keys/{key_name}")
        return ApiKeyResponse(**data)

    async def update_api_key(self, key_name: str, api_key_update: ApiKeyUpdate) -> ApiKeyResponse:
        """
        Update an API key.

        Args:
            key_name (str): The name of the API key to update.
            api_key_update (ApiKeyUpdate): The updated API key information.

        Returns:
            ApiKeyResponse: The updated API key information.
        """
        self.logger.info(f"Updating API key: {key_name}")
        data = await self._request("PATCH", f"/api-keys/{key_name}", json=api_key_update.dict(exclude_unset=True))
        return ApiKeyResponse(**data)

    async def revoke_api_key(self, key_name: str) -> ApiKeyResponse:
        """
        Revoke an API key.

        Args:
            key_name (str): The name of the API key to revoke.

        Returns:
            ApiKeyResponse: The revoked API key information.
        """
        self.logger.info(f"Revoking API key: {key_name}")
        data = await self._request("DELETE", f"/api-keys/{key_name}")
        return ApiKeyResponse(**data)

    # Datasets methods

    async def upload_dataset(self, file_path: str, dataset_create: DatasetCreate) -> DatasetResponse:
        """
        Upload a new dataset.

        Args:
            file_path (str): The path to the dataset file.
            dataset_create (DatasetCreate): The dataset creation data.

        Returns:
            DatasetResponse: The uploaded dataset information.
        """
        self.logger.info(f"Uploading dataset: {dataset_create.name}")
        data = aiohttp.FormData()
        data.add_field('file', open(file_path, 'rb'))
        data.add_field('name', dataset_create.name)
        if dataset_create.description:
            data.add_field('description', dataset_create.description)
        data = await self._request("POST", "/datasets", data=data)
        return DatasetResponse(**data)

    async def list_datasets(self, page: int = 1, items_per_page: int = 20) -> ListResponse:
        """
        List all datasets.

        Args:
            page (int): The page number to retrieve. Defaults to 1.
            items_per_page (int): The number of items per page. Defaults to 20.

        Returns:
            ListResponse: A list of datasets and pagination information.
        """
        self.logger.info(f"Listing datasets (page {page})")
        params = {"page": page, "items_per_page": items_per_page}
        data = await self._request("GET", "/datasets", params=params)
        return ListResponse(
            data=[DatasetResponse(**item) for item in data['data']],
            pagination=Pagination(**data['pagination'])
        )

    async def get_dataset(self, dataset_name: str) -> DatasetResponse:
        """
        Get information about a specific dataset.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            DatasetResponse: The dataset information.
        """
        self.logger.info(f"Getting dataset: {dataset_name}")
        data = await self._request("GET", f"/datasets/{dataset_name}")
        return DatasetResponse(**data)

    async def update_dataset(self, dataset_name: str, dataset_update: DatasetUpdate) -> DatasetResponse:
        """
        Update a dataset.

        Args:
            dataset_name (str): The name of the dataset to update.
            dataset_update (DatasetUpdate): The updated dataset information.

        Returns:
            DatasetResponse: The updated dataset information.
        """
        self.logger.info(f"Updating dataset: {dataset_name}")
        data = await self._request("PATCH", f"/datasets/{dataset_name}", json=dataset_update.dict(exclude_unset=True))
        return DatasetResponse(**data)

    async def delete_dataset(self, dataset_name: str) -> None:
        """
        Delete a dataset.

        Args:
            dataset_name (str): The name of the dataset to delete.
        """
        self.logger.info(f"Deleting dataset: {dataset_name}")
        await self._request("DELETE", f"/datasets/{dataset_name}")

    # Fine-tuning Jobs methods

    async def create_fine_tuning_job(self, job_create: FineTuningJobCreate) -> FineTuningJobResponse:
        """
        Create a new fine-tuning job.

        Args:
            job_create (FineTuningJobCreate): The fine-tuning job creation data.

        Returns:
            FineTuningJobResponse: The created fine-tuning job information.
        """
        self.logger.info(f"Creating fine-tuning job: {job_create.name}")
        data = await self._request("POST", "/fine-tuning", json=job_create.dict())
        return FineTuningJobResponse(**data)

    async def list_fine_tuning_jobs(self, page: int = 1, items_per_page: int = 20) -> ListResponse:
        """
        List all fine-tuning jobs.

        Args:
            page (int): The page number to retrieve. Defaults to 1.
            items_per_page (int): The number of items per page. Defaults to 20.

        Returns:
            ListResponse: A list of fine-tuning jobs and pagination information.
        """
        self.logger.info(f"Listing fine-tuning jobs (page {page})")
        params = {"page": page, "items_per_page": items_per_page}
        data = await self._request("GET", "/fine-tuning", params=params)
        return ListResponse(
            data=[FineTuningJobResponse(**item) for item in data['data']],
            pagination=Pagination(**data['pagination'])
        )

    async def get_fine_tuning_job(self, job_name: str) -> FineTuningJobDetailResponse:
        """
        Get detailed information about a specific fine-tuning job.

        Args:
            job_name (str): The name of the fine-tuning job.

        Returns:
            FineTuningJobDetailResponse: Detailed information about the fine-tuning job.
        """
        self.logger.info(f"Getting fine-tuning job: {job_name}")
        data = await self._request("GET", f"/fine-tuning/{job_name}")
        return FineTuningJobDetailResponse(**data)

    async def cancel_fine_tuning_job(self, job_name: str) -> FineTuningJobDetailResponse:
        """
        Cancel a fine-tuning job.

        Args:
            job_name (str): The name of the fine-tuning job to cancel.

        Returns:
            FineTuningJobDetailResponse: Updated information about the cancelled fine-tuning job.
        """
        self.logger.info(f"Cancelling fine-tuning job: {job_name}")
        data = await self._request("POST", f"/fine-tuning/{job_name}/cancel")
        return FineTuningJobDetailResponse(**data)

    # Models methods

    async def list_base_models(self, page: int = 1, items_per_page: int = 20) -> ListResponse:
        """
        List all available base models.

        Args:
            page (int): The page number to retrieve. Defaults to 1.
            items_per_page (int): The number of items per page. Defaults to 20.

        Returns:
            ListResponse: A list of base models and pagination information.
        """
        self.logger.info(f"Listing base models (page {page})")
        params = {"page": page, "items_per_page": items_per_page}
        data = await self._request("GET", "/models/base", params=params)
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
        """
        self.logger.info(f"Getting base model: {model_name}")
        data = await self._request("GET", f"/models/base/{model_name}")
        return BaseModelResponse(**data)

    async def list_fine_tuned_models(self, page: int = 1, items_per_page: int = 20) -> ListResponse:
        """
        List all fine-tuned models.

        Args:
            page (int): The page number to retrieve. Defaults to 1.
            items_per_page (int): The number of items per page. Defaults to 20.

        Returns:
            ListResponse: A list of fine-tuned models and pagination information.
        """
        self.logger.info(f"Listing fine-tuned models (page {page})")
        params = {"page": page, "items_per_page": items_per_page}
        data = await self._request("GET", "/models/fine-tuned", params=params)
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
        """
        self.logger.info(f"Getting fine-tuned model: {model_name}")
        data = await self._request("GET", f"/models/fine-tuned/{model_name}")
        return FineTunedModelResponse(**data)

        # Usage methods

    async def get_total_cost(self, start_date: datetime, end_date: datetime) -> TotalCostResponse:
        """
        Get the total cost for a given period.

        Args:
            start_date (datetime): The start date of the period.
            end_date (datetime): The end date of the period.

        Returns:
            TotalCostResponse: The total cost information for the specified period.
        """
        self.logger.info(f"Getting total cost from {start_date} to {end_date}")
        params = {
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat()
        }
        data = await self._request("GET", "/usage/total-cost", params=params)
        return TotalCostResponse(**data)

    async def list_usage_records(self, start_date: datetime, end_date: datetime, page: int = 1, items_per_page: int = 20) -> ListResponse:
        """
        List usage records for a given period.

        Args:
            start_date (datetime): The start date of the period.
            end_date (datetime): The end date of the period.
            page (int): The page number to retrieve. Defaults to 1.
            items_per_page (int): The number of items per page. Defaults to 20.

        Returns:
            ListResponse: A list of usage records and pagination information.
        """
        self.logger.info(f"Listing usage records from {start_date} to {end_date} (page {page})")
        params = {
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "page": page,
            "items_per_page": items_per_page
        }
        data = await self._request("GET", "/usage/records", params=params)
        return ListResponse(
            data=[UsageRecordResponse(**item) for item in data['data']],
            pagination=Pagination(**data['pagination'])
        )
