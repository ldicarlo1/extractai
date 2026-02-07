# AGENTS.md

Instructions for automated coding agents working in this repository.

## Scope
- This file applies to the entire repository.

## Project Overview
- Package name: `extractai`
- Python version: `>=3.11`
- Packaging source of truth: `pyproject.toml`
- Main library code: `extractai/core.py`
- Frontend app: `extractai/extractai-app.py`
- Tests: `tests/test_core.py`
- Demo notebooks:
  - `extractai_basic_demo.ipynb`
  - `extractai_batch_demo.ipynb`

## Non-Negotiable Rules
- Do not reintroduce `extractify` package/module names.
- Keep public imports using `extractai` only.
- Preserve CSV columns and order:
  - `id`, schema fields, `model`, `status`, `input_tokens`, `output_tokens`
- Keep status semantics stable:
  - `COMPLETE`, `FAILED`, `SKIPPED`
- If a single document fails, processing must continue for remaining documents.

## Development Workflow
1. Make focused, minimal changes.
2. Run the validation pipeline before finishing:
   - `./scripts/test_pipeline.sh`
3. If API behavior changes, update:
   - `README.md`
   - notebooks
   - tests

## Environment Notes
- API key is read from `OPENAI_API_KEY`.
- `.env.example` is provided; never commit real secrets.

## Batch MVP Constraints
- Batch endpoint currently uses `/v1/chat/completions`.
- Completion window must remain `24h` unless explicitly redesigned.
- Batch artifacts are stored under:
  - `outputs/.extractai/batches/<submission_id>/`

## Packaging and Release
- Build/publish metadata lives in `pyproject.toml` and `.github/workflows/publish.yml`.
- Ensure `README.md` install and usage examples match the current API (`ExtractAIConfig`).
