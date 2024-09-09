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
pip install "git+ssh://git@github.com/luminolabs/lumino-sdk-python.git"
```

## Quick Start

Here's a simple example of how to use the Lumino SDK. More examples under `tests/e2e_test.py`.

```python
import asyncio
from lumino.sdk import LuminoSDK

async def main():
    async with LuminoSDK("your-api-key") as sdk:
        user = await sdk.user.get_current_user()

asyncio.run(main())
```


## Support

If you encounter any problems or have any questions, please open an issue on the GitHub repository.