# Lumino SDK for Python

The Lumino SDK for Python provides a convenient way to interact with the Lumino API for managing large language model (LLM) fine-tuning processes and related resources.

## Features

- User management
- API key management
- Dataset operations
- Fine-tuning job management
- Model information retrieval
- Usage tracking

## Installation

You can install the Lumino SDK using pip:

```bash
pip install lumino-sdk
```

## Quick Start

Here's a simple example of how to use the Lumino SDK:

```python
import asyncio
from lumino.sdk import LuminoSDK
from lumino.models import DatasetCreate

async def main():
    async with LuminoSDK("your-api-key") as sdk:
        # Get current user info
        user = await sdk.get_current_user()
        print(f"Current user: {user.name}")

        # Upload a dataset
        dataset = await sdk.upload_dataset(
            "path/to/your/dataset.jsonl",
            DatasetCreate(name="my_dataset", description="A test dataset")
        )
        print(f"Uploaded dataset: {dataset.name}")

        # List fine-tuning jobs
        jobs = await sdk.list_fine_tuning_jobs()
        print(f"You have {len(jobs.data)} fine-tuning jobs")

asyncio.run(main())
```


## Support

If you encounter any problems or have any questions, please open an issue on the GitHub repository.