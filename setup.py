from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("VERSION", "r", encoding="utf-8") as fh:
    version = fh.read()

setup(
    name="lumino",
    version=version,
    author="Lumino Labs AI",
    author_email="engg@luminolabs.ai",
    license="Apache License 2.0",
    description="A Python SDK for interacting with the Lumino Labs API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luminolabs/lumino-sdk-python",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "aiohttp",
        "pydantic",
        "pydantic[email]",
    ],
)