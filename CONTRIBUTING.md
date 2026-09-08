# Contributing to PINNeAPPle

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

## Development history and internal engineering notes
[`docs/dev/ROADMAP_PHYSICS_AI_HUB.md`](docs/dev/ROADMAP_PHYSICS_AI_HUB.md) and
[`docs/dev/AUDIT_REPORT.md`](docs/dev/AUDIT_REPORT.md) are internal engineering
notes kept for historical context: a running roadmap and an audit log written
during development sessions. They are not polished, curated documentation —
expect first-person session narration, in-progress task tracking, and
references to work that may since have changed — but they're useful if you
want the backstory on why a design decision was made or what's already been
tried. For current, user-facing documentation see the [`docs/`](docs/)
directory and [`README.md`](README.md).
