# dipole-utils

## Testing

Run the default test suite (skipping slow opt-in tests) with:

```bash
pytest
```

Tests marked with `@pytest.mark.slow` are disabled by default via `pytest.ini`. To include them:

- Only slow tests:

  ```bash
  pytest -m slow
  ```

- All tests, including slow ones:

  ```bash
  pytest -m "slow or not slow"
  ```
