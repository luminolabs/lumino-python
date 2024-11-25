import asyncio
import logging
import os
import random
import re
from datetime import datetime, timedelta, date
from typing import Any

from lumino.models import (
    UserUpdate,
    ApiKeyCreate,
    ApiKeyUpdate,
    DatasetCreate,
    DatasetUpdate,
    FineTuningJobCreate,
    FineTuningJobParameters,
    FineTuningJobType, ApiKeyStatus
)
from lumino.sdk import LuminoSDK

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('e2e_test')

# Configuration
API_KEY = os.environ.get("LUMSDK_API_KEY", "your-api-key")
BASE_URL = os.environ.get("LUMSDK_BASE_URL", "http://localhost:5100/v1")
RANDOM_SUFFIX = f"{random.randint(1000, 9999):04d}"


def add_suffix(name: str) -> str:
    """Add a random suffix to a name and ensure it follows naming pattern."""
    name = re.sub(r'[^a-z0-9-]', '-', name.lower())
    name = re.sub(r'^-+|-+$', '', name)
    name = re.sub(r'-+', '-', name)
    return f"{name}-{RANDOM_SUFFIX}"


class E2ETest:
    """
    Comprehensive end-to-end test class for Lumino SDK.
    Includes test methods for all API operations and helper methods for test validation.
    """

    def __init__(self):
        """Initialize test instance with SDK client."""
        self.sdk = LuminoSDK(API_KEY, BASE_URL)
        self.test_data = {}

    async def __aenter__(self):
        """Set up test environment."""
        await self.sdk.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up test environment."""
        await self.sdk.__aexit__(exc_type, exc_val, exc_tb)

    def assert_equal(self, actual: Any, expected: Any, message: str):
        """Assert equality with logging."""
        assert actual == expected, f"{message}: expected {expected}, got {actual}"
        logger.info(f"✓ {message}")

    async def test_user_operations(self):
        """Test user management operations."""
        logger.info("\n=== Testing User Operations ===")

        # Get current user
        user = await self.sdk.user.get_current_user()
        logger.info(f"Current user: {user.email}")
        self.test_data['original_name'] = user.name

        # Update user
        new_name = add_suffix("test-user")
        updated_user = await self.sdk.user.update_current_user(UserUpdate(name=new_name))
        self.assert_equal(updated_user.name, new_name, "User name updated")

        # Restore original name
        await self.sdk.user.update_current_user(UserUpdate(name=self.test_data['original_name']))

    async def test_api_key_operations(self):
        """Test API key management operations."""
        logger.info("\n=== Testing API Key Operations ===")

        # Create API key
        key_name = add_suffix("test-key")
        new_key = await self.sdk.api_keys.create_api_key(
            ApiKeyCreate(
                name=key_name,
                expires_at=datetime.now() + timedelta(days=30)
            )
        )
        self.assert_equal(new_key.name, key_name, "API key created")
        self.test_data['api_key'] = new_key

        # List API keys
        keys = await self.sdk.api_keys.list_api_keys()
        self.assert_equal(
            any(k.id == new_key.id for k in keys.data),
            True,
            "New API key found in list"
        )

        # Update API key
        updated_name = add_suffix("updated-key")
        updated_key = await self.sdk.api_keys.update_api_key(
            new_key.name,
            ApiKeyUpdate(name=updated_name)
        )
        self.assert_equal(updated_key.name, updated_name, "API key name updated")

        # Revoke API key
        revoked_key = await self.sdk.api_keys.revoke_api_key(updated_name)
        self.assert_equal(revoked_key.status, ApiKeyStatus.REVOKED, "API key revoked")

    async def test_dataset_operations(self):
        """Test dataset management operations."""
        logger.info("\n=== Testing Dataset Operations ===")

        # Upload dataset
        dataset_name = add_suffix("test-dataset")
        dataset = await self.sdk.dataset.upload_dataset(
            "artifacts/e2e_test.jsonl",
            DatasetCreate(
                name=dataset_name,
                description="Test dataset for E2E testing"
            )
        )
        self.assert_equal(dataset.name, dataset_name, "Dataset uploaded")
        self.test_data['dataset'] = dataset

        # List datasets
        datasets = await self.sdk.dataset.list_datasets()
        self.assert_equal(
            any(d.id == dataset.id for d in datasets.data),
            True,
            "New dataset found in list"
        )

        # Get dataset details
        dataset_info = await self.sdk.dataset.get_dataset(dataset.name)
        self.assert_equal(dataset_info.id, dataset.id, "Dataset details retrieved")

        # Update dataset
        new_description = "Updated test dataset description"
        updated_dataset = await self.sdk.dataset.update_dataset(
            dataset.name,
            DatasetUpdate(description=new_description)
        )
        self.assert_equal(updated_dataset.description, new_description, "Dataset updated")

        # Delete dataset
        await self.sdk.dataset.delete_dataset(dataset.name)
        logger.info("✓ Dataset deleted")

    async def test_model_operations(self):
        """Test model operations."""
        logger.info("\n=== Testing Model Operations ===")

        # List base models
        base_models = await self.sdk.model.list_base_models()
        self.assert_equal(len(base_models.data) > 0, True, "Base models listed")
        self.test_data['base_model'] = base_models.data[0]

        # Get base model details
        model_details = await self.sdk.model.get_base_model(base_models.data[0].name)
        self.assert_equal(model_details.id, base_models.data[0].id, "Base model details retrieved")

        # List fine-tuned models
        fine_tuned_models = await self.sdk.model.list_fine_tuned_models()
        logger.info(f"Found {len(fine_tuned_models.data)} fine-tuned models")

    async def test_fine_tuning_operations(self):
        """Test fine-tuning job operations."""
        logger.info("\n=== Testing Fine-tuning Operations ===")

        # Upload dataset
        dataset_name = add_suffix("test-dataset-2")
        dataset = await self.sdk.dataset.upload_dataset(
            "artifacts/e2e_test.jsonl",
            DatasetCreate(
                name=dataset_name,
                description="Test dataset for E2E testing"
            )
        )

        # Create fine-tuning job
        job_name = add_suffix("test-job")
        job = await self.sdk.fine_tuning.create_fine_tuning_job(
            FineTuningJobCreate(
                base_model_name=self.test_data['base_model'].name,
                dataset_name=dataset.name,
                name=job_name,
                type=FineTuningJobType.LORA,
                parameters=FineTuningJobParameters(
                    batch_size=2,
                    shuffle=True,
                    num_epochs=1,
                )
            )
        )
        self.assert_equal(job.name, job_name, "Fine-tuning job created")
        self.test_data['job'] = job

        # List fine-tuning jobs
        jobs = await self.sdk.fine_tuning.list_fine_tuning_jobs()
        self.assert_equal(
            any(j.id == job.id for j in jobs.data),
            True,
            "New job found in list"
        )

        # Get job details
        job_details = await self.sdk.fine_tuning.get_fine_tuning_job(job.name)
        self.assert_equal(job_details.id, job.id, "Job details retrieved")

        # Delete job
        await self.sdk.fine_tuning.delete_fine_tuning_job(job.name)
        logger.info("✓ Fine-tuning job deleted")

        # Delete dataset
        await self.sdk.dataset.delete_dataset(dataset.name)
        logger.info("✓ Dataset deleted")

    async def test_usage_and_billing(self):
        """Test usage tracking and billing operations."""
        logger.info("\n=== Testing Usage and Billing Operations ===")

        # Set date range for last 30 days
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        # Get total cost
        total_cost = await self.sdk.usage.get_total_cost(start_date, end_date)
        logger.info(f"✓ Total cost retrieved: ${total_cost.total_cost:.2f}")

        # List usage records
        usage_records = await self.sdk.usage.list_usage_records(start_date, end_date)
        logger.info(f"✓ Retrieved {len(usage_records.data)} usage records")

        # Get credit history
        credit_history = await self.sdk.billing.get_credit_history(start_date, end_date)
        logger.info(f"✓ Retrieved {len(credit_history.data)} credit history records")

    async def test_admin_operations(self):
        """Test operations that require admin privileges."""
        logger.info("\n=== Testing Admin Operations ===")

        # Get current user
        current_user = await self.sdk.user.get_current_user()

        # Create a test job to use for credit deduction
        job_name = None
        dataset_name = None

        try:
            # Create test dataset first
            dataset_name = add_suffix("admin-test-dataset")
            dataset = await self.sdk.dataset.upload_dataset(
                "artifacts/e2e_test.jsonl",
                DatasetCreate(
                    name=dataset_name,
                    description="Test dataset for admin testing"
                )
            )
            logger.info("✓ Created test dataset for admin operations")

            # Get first available base model
            base_models = await self.sdk.model.list_base_models()
            base_model = base_models.data[0]

            # Create a test fine-tuning job
            job_name = add_suffix("admin-test-job")
            job = await self.sdk.fine_tuning.create_fine_tuning_job(
                FineTuningJobCreate(
                    base_model_name=base_model.name,
                    dataset_name=dataset.name,
                    name=job_name,
                    type=FineTuningJobType.LORA,
                    parameters=FineTuningJobParameters(
                        batch_size=2,
                        shuffle=True,
                        num_epochs=1,
                    )
                )
            )
            logger.info("✓ Created test job for admin operations")

            # Test credit deduction endpoint
            deduct_request = {
                "user_id": str(current_user.id),
                "usage_amount": 1000000,  # 1M tokens
                "usage_unit": "TOKEN",
                "service_name": "FINE_TUNING_JOB",
                "fine_tuning_job_id": str(job.id)
            }
            response = await self.sdk._request(
                "POST",
                "/billing/credits-deduct",
                json=deduct_request
            )
            self.assert_equal(
                'transaction_type' in response,
                True,
                "Credit deduction endpoint responded"
            )
            logger.info("✓ Credit deduction endpoint tested")

            # Test credit addition endpoint
            add_request = {
                "user_id": str(current_user.id),
                "amount": 50.0,
                "transaction_id": f"test-credit-{RANDOM_SUFFIX}"
            }
            response = await self.sdk._request(
                "POST",
                "/billing/credits-add",
                json=add_request
            )
            self.assert_equal(
                'transaction_type' in response,
                True,
                "Credit addition endpoint responded"
            )
            logger.info("✓ Credit addition endpoint tested")

        except Exception as e:
            logger.error(f"Admin operations test failed: {str(e)}")
            # Log additional error details if available
            if hasattr(e, 'details'):
                logger.error(f"Error details: {e.details}")
            raise

        finally:
            # Clean up test resources
            try:
                if job_name:
                    await self.sdk.fine_tuning.delete_fine_tuning_job(job_name)
                    logger.info("✓ Cleaned up test job")

                if dataset_name:
                    await self.sdk.dataset.delete_dataset(dataset_name)
                    logger.info("✓ Cleaned up test dataset")

            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed: {str(cleanup_error)}")

    async def run_all_tests(self):
        """Run all test cases in sequence."""
        try:
            await self.test_user_operations()
            await self.test_api_key_operations()
            await self.test_dataset_operations()
            await self.test_model_operations()
            await self.test_fine_tuning_operations()
            await self.test_usage_and_billing()
            await self.test_admin_operations()
            logger.info("\n=== All tests completed successfully! ===")
        except Exception as e:
            logger.error(f"\n!!! Test failed: {str(e)}")
            raise


async def main():
    """Main entry point for running the E2E tests."""
    async with E2ETest() as test:
        await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
