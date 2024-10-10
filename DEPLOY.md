### Deploying to PyPI

1. Create an API key on PyPI
2. Create a `.pypirc` file in your home directory with the following content:
```bash
[distutils]
  index-servers =
    lumino

[lumino]
  repository = https://upload.pypi.org/legacy/
  username = __token__
  password = <API_KEY>
```
    
3. Make sure the `VERSION` file is updated
4. Run the following command:
```bash
./scripts/push-to-pypi.sh
```