import asyncio
import logging
import os
import random
import re
from datetime import datetime, timedelta, date

from lumino.models import (
    UserUpdate, ApiKeyCreate, DatasetCreate, FineTuningJobCreate,
    ApiKeyUpdate, DatasetUpdate, FineTuningJobParameters, FineTuningJobType
)
from lumino.sdk import LuminoSDK

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('e2e_test')

# Get API key and base URL from environment variables
API_KEY = os.environ.get("LUMSDK_API_KEY")
BASE_URL = os.environ.get("LUMSDK_BASE_URL", "https://api.luminolabs.ai/v1")

# Generate a random 4-digit suffix for unique naming
RANDOM_SUFFIX = f"{random.randint(1000, 9999):04d}"


def add_suffix(name: str) -> str:
    """
    Add a random suffix to a name for uniqueness and ensure it follows ^[a-z0-9-]+$ pattern.

    Args:
        name (str): The original name.

    Returns:
        str: The name with a random suffix added, following the required pattern.
    """
    # Convert to lowercase and replace non-alphanumeric characters with hyphens
    name = re.sub(r'[^a-z0-9-]', '-', name.lower())
    # Remove leading/trailing hyphens and collapse multiple hyphens
    name = re.sub(r'^-+|-+$', '', name)
    name = re.sub(r'-+', '-', name)
    return f"{name}-{RANDOM_SUFFIX}"


async def run_e2e_test():
    """
    Run an end-to-end test of the Lumino SDK.

    This function tests various functionalities of the SDK including user management,
    API key management, dataset operations, fine-tuning job operations, model operations,
    and usage tracking.
    """
    async with LuminoSDK(API_KEY, BASE_URL) as sdk:
        # Test User Management
        logger.info("Testing User Management")
        user = await sdk.user.get_current_user()
        logger.info(f"Current user: {user.name} ({user.email})")

        updated_user = await sdk.user.update_current_user(UserUpdate(name=f"Updated User {RANDOM_SUFFIX}"))
        logger.info(f"Updated user name: {updated_user.name}")

        # Test API Key Management
        logger.info("\nTesting API Key Management")
        new_api_key = await sdk.api_keys.create_api_key(ApiKeyCreate(name=add_suffix("test-key"), expires_at=datetime.now() + timedelta(days=30)))
        logger.info(f"Created new API key: {new_api_key.name} (Secret: {new_api_key.secret})")

        api_keys = await sdk.api_keys.list_api_keys()
        logger.info(f"Listed {len(api_keys.data)} API keys")

        updated_api_key = await sdk.api_keys.update_api_key(new_api_key.name, ApiKeyUpdate(name=add_suffix("updated-key")))
        logger.info(f"Updated API key name: {updated_api_key.name}")

        revoked_api_key = await sdk.api_keys.revoke_api_key(updated_api_key.name)
        logger.info(f"Revoked API key: {revoked_api_key.name} (Status: {revoked_api_key.status})")

        # Test Dataset Operations
        logger.info("\nTesting Dataset Operations")
        dataset = await sdk.dataset.upload_dataset("artifacts/e2e_test.jsonl", DatasetCreate(name=add_suffix("test-dataset"), description="a-test-dataset"))
        logger.info(f"Uploaded dataset: {dataset.name} (ID: {dataset.id})")

        datasets = await sdk.dataset.list_datasets()
        logger.info(f"Listed {len(datasets.data)} datasets")

        dataset_info = await sdk.dataset.get_dataset(dataset.name)
        logger.info(f"Retrieved dataset info: {dataset_info.name} (Status: {dataset_info.status})")

        updated_dataset = await sdk.dataset.update_dataset(dataset.name, DatasetUpdate(description=add_suffix("updated-description")))
        logger.info(f"Updated dataset description: {updated_dataset.description}")

        # Test Fine-tuning Job Operations
        logger.info("\nTesting Fine-tuning Job Operations")

        # List available base models
        base_models = await sdk.model.list_base_models()
        logger.info(f"Listed {len(base_models.data)} base models")

        if base_models.data:
            selected_base_model = base_models.data[0].name
            logger.info(f"Selected base model: {selected_base_model}")

            job = await sdk.fine_tuning.create_fine_tuning_job(FineTuningJobCreate(
                base_model_name=selected_base_model,
                dataset_name=dataset.name,
                name=add_suffix("test-fine-tuning-job"),
                type=FineTuningJobType.LORA,
                parameters=FineTuningJobParameters(
                    batch_size=2,
                    shuffle=True,
                    num_epochs=1,
                )
            ))
            logger.info(f"Created fine-tuning job: {job.name} (ID: {job.id})")

            jobs = await sdk.fine_tuning.list_fine_tuning_jobs()
            logger.info(f"Listed {len(jobs.data)} fine-tuning jobs")

            job_details = await sdk.fine_tuning.get_fine_tuning_job(job.name)
            logger.info(f"Retrieved job details: {job_details.name} (Status: {job_details.status})")

            # Uncomment the following lines to test job cancellation
            # cancelled_job = await sdk.fine_tuning.cancel_fine_tuning_job(job.name)
            # logger.info(f"Cancelled fine-tuning job: {cancelled_job.name} (Status: {cancelled_job.status})")
        else:
            logger.warning("No base models available. Skipping fine-tuning job creation.")

        # Test Model Operations
        logger.info("\nTesting Model Operations")
        base_models = await sdk.model.list_base_models()
        logger.info(f"Listed {len(base_models.data)} base models")

        if base_models.data:
            base_model = await sdk.model.get_base_model(base_models.data[0].name)
            logger.info(f"Retrieved base model: {base_model.name} (Status: {base_model.status})")

        fine_tuned_models = await sdk.model.list_fine_tuned_models()
        logger.info(f"Listed {len(fine_tuned_models.data)} fine-tuned models")

        if fine_tuned_models.data:
            fine_tuned_model = await sdk.model.get_fine_tuned_model(fine_tuned_models.data[0].name)
            logger.info(f"Retrieved fine-tuned model: {fine_tuned_model.name}")

        # Test Usage Tracking
        logger.info("\nTesting Usage Tracking")
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        total_cost = await sdk.usage.get_total_cost(start_date, end_date)
        logger.info(f"Total cost from {start_date} to {end_date}: ${total_cost.total_cost:.2f}")

        usage_records = await sdk.usage.list_usage_records(start_date, end_date)
        logger.info(f"Listed {len(usage_records.data)} usage records")

        # Test Credits
        credits = await sdk.billing.get_credit_history(start_date, end_date)
        logger.info(f"Credit history from {start_date} to {end_date}: {credits.data}")

        logger.info("\nEnd-to-end test completed successfully!")

if __name__ == "__main__":
    asyncio.run(run_e2e_test())
