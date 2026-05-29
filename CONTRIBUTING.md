# Contributing to Pinneapple

Thanks for taking the time to contribute!

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -q
```

## Lint / format

```bash
ruff check .
ruff format .
```

## Pull requests
- Keep changes focused and well-scoped
- Add or update tests when possible
- Update docs/examples if behavior changes

## Commit style
We recommend Conventional Commits (optional), e.g.:
- feat: add shard-aware iterator
- fix: correct zarr cache eviction
- docs: improve README examples
