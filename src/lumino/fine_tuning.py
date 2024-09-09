"""
Fine-tuning job management endpoints for the Lumino SDK.

This module contains the FineTuningEndpoint class, which provides methods
for interacting with fine-tuning job-related API endpoints.
"""

import logging
from typing import Any, List, Optional

from lumino.models import (
    FineTuningJobCreate,
    FineTuningJobResponse,
    FineTuningJobDetailResponse,
    ListResponse,
    Pagination
)


class FineTuningEndpoint:
    """
    Handles fine-tuning job-related API endpoints for the Lumino SDK.
    """

    def __init__(self, sdk: Any):
        """
        Initialize the FineTuningEndpoint.

        Args:
            sdk (Any): The LuminoSDK instance.
        """
        self._sdk = sdk
        self.logger = logging.getLogger(__name__)

    async def create_fine_tuning_job(self, job_create: FineTuningJobCreate) -> FineTuningJobResponse:
        """
        Create a new fine-tuning job.

        Args:
            job_create (FineTuningJobCreate): The fine-tuning job creation data.

        Returns:
            FineTuningJobResponse: The created fine-tuning job information.

        Raises:
            LuminoAPIError: If the API request fails.
            LuminoValidationError: If the provided data is invalid.
        """
        self.logger.info("Creating fine-tuning job: %s", job_create.name)
        data = await self._sdk._request("POST", "/fine-tuning", json=job_create.dict())
        return FineTuningJobResponse(**data)

    async def list_fine_tuning_jobs(self, page: int = 1, items_per_page: int = 20,
                                    status: Optional[str] = None) -> ListResponse:
        """
        List all fine-tuning jobs.

        Args:
            page (int): The page number to retrieve. Defaults to 1.
            items_per_page (int): The number of items per page. Defaults to 20.
            status (Optional[str]): Filter jobs by status. If None, all jobs are returned.

        Returns:
            ListResponse: A list of fine-tuning jobs and pagination information.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Listing fine-tuning jobs (page %d)", page)
        params = {"page": page, "items_per_page": items_per_page}
        if status:
            params["status"] = status
        data = await self._sdk._request("GET", "/fine-tuning", params=params)
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

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Getting fine-tuning job: %s", job_name)
        data = await self._sdk._request("GET", f"/fine-tuning/{job_name}")
        return FineTuningJobDetailResponse(**data)

    async def cancel_fine_tuning_job(self, job_name: str) -> FineTuningJobDetailResponse:
        """
        Cancel a fine-tuning job.

        Args:
            job_name (str): The name of the fine-tuning job to cancel.

        Returns:
            FineTuningJobDetailResponse: Updated information about the cancelled fine-tuning job.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Cancelling fine-tuning job: %s", job_name)
        data = await self._sdk._request("POST", f"/fine-tuning/{job_name}/cancel")
        return FineTuningJobDetailResponse(**data)

    async def delete_fine_tuning_job(self, job_name: str) -> None:
        """
        Delete a fine-tuning job.

        Args:
            job_name (str): The name of the fine-tuning job to delete.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Deleting fine-tuning job: %s", job_name)
        await self._sdk._request("DELETE", f"/fine-tuning/{job_name}")

    async def get_fine_tuning_job_metrics(self, job_name: str) -> dict:
        """
        Get metrics for a specific fine-tuning job.

        Args:
            job_name (str): The name of the fine-tuning job.

        Returns:
            dict: A dictionary containing metrics for the fine-tuning job.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Getting metrics for fine-tuning job: %s", job_name)
        data = await self._sdk._request("GET", f"/fine-tuning/{job_name}/metrics")
        return data

    async def get_fine_tuning_job_logs(self, job_name: str, start_time: Optional[str] = None,
                                       end_time: Optional[str] = None) -> List[str]:
        """
        Get logs for a specific fine-tuning job.

        Args:
            job_name (str): The name of the fine-tuning job.
            start_time (Optional[str]): The start time for log retrieval (ISO format).
            end_time (Optional[str]): The end time for log retrieval (ISO format).

        Returns:
            List[str]: A list of log entries for the fine-tuning job.

        Raises:
            LuminoAPIError: If the API request fails.
        """
        self.logger.info("Getting logs for fine-tuning job: %s", job_name)
        params = {}
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        data = await self._sdk._request("GET", f"/fine-tuning/{job_name}/logs", params=params)
        return data["logs"]
