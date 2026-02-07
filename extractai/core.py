from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generic, Sequence, TypeVar
from uuid import uuid4

from openai import OpenAI
from pydantic import BaseModel

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]

SchemaT = TypeVar("SchemaT", bound=BaseModel)

DEFAULT_MAX_INPUT_TOKENS = 50_000
STATUS_COMPLETE = "COMPLETE"
STATUS_FAILED = "FAILED"
STATUS_SKIPPED = "SKIPPED"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}
BATCH_STATUS_FILE_NAME = "batch_status.json"
BATCH_OUTPUT_FILE_NAME = "output.jsonl"
BATCH_ERROR_FILE_NAME = "error.jsonl"


class InputTokenLimitExceeded(ValueError):
    """Raised when input tokens exceed the configured limit."""

    def __init__(self, *, file_name: str, total_input_tokens: int, max_tokens: int) -> None:
        self.file_name = file_name
        self.total_input_tokens = total_input_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"Input for {file_name} is {total_input_tokens} tokens, "
            f"which exceeds the {max_tokens} token limit."
        )


@dataclass(slots=True)
class ExtractAIConfig:
    """Runtime options for extraction."""

    pdf_dir: str | Path
    model: str = "gpt-5-nano"
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    use_batch: bool = False
    api_key: str | None = None


@dataclass(slots=True)
class ExtractionResult(Generic[SchemaT]):
    """Extraction result for one document."""

    file_name: str
    pdf_path: Path
    input_tokens: int | None
    output: SchemaT
    output_tokens: int | None = None
    model: str | None = None
    status: str = STATUS_COMPLETE
    skipped: bool = False
    error: str | None = None


@dataclass(slots=True)
class BatchSubmission:
    """Submission metadata for a batch run."""

    submission_id: str
    batch_id: str | None
    input_file_id: str | None
    batch_dir: Path
    manifest_path: Path
    input_jsonl_path: Path
    submitted_requests: int
    skipped_requests: int
    failed_requests: int
    status: str


@dataclass(slots=True)
class _PreparedDocument:
    path: Path
    user_content: str
    input_tokens: int


def build_prompt(prompt: str) -> str:
    """Return a trimmed, non-empty prompt."""

    cleaned = prompt.strip()
    if not cleaned:
        raise ValueError("prompt cannot be empty")
    return cleaned


def _ensure_positive_max_input_tokens(value: int) -> None:
    if value <= 0:
        raise ValueError("max_input_tokens must be > 0")


def list_pdf_files(pdf_dir: str | Path) -> list[Path]:
    """List PDFs in sorted order."""

    directory = Path(pdf_dir)
    files = sorted(directory.glob("*.pdf"))
    if not files:
        raise FileNotFoundError(f"No PDFs found in {directory.resolve()}")
    return files


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract plain text from all pages in a PDF."""

    if pdfplumber is None:
        raise ModuleNotFoundError(
            "pdfplumber is required for PDF extraction. "
            "Install dependencies with: python -m pip install -e .[demo]"
        )

    path = Path(pdf_path)
    with pdfplumber.open(path) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages).strip()


def make_token_counter(model: str) -> Callable[[str], int]:
    """Return a token counter for the given model."""

    try:
        import tiktoken

        try:
            encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            encoder = tiktoken.get_encoding("o200k_base")

        return lambda text: len(encoder.encode(text))
    except Exception:
        return lambda text: max(1, len(text) // 4)


def ensure_input_token_limit(
    *,
    base_prompt: str,
    user_content: str,
    max_tokens: int,
    file_name: str,
    count_tokens: Callable[[str], int],
) -> int:
    """Return total input tokens if under limit, otherwise raise."""

    total_input_tokens = count_tokens(base_prompt) + count_tokens(user_content)
    if total_input_tokens >= max_tokens:
        raise InputTokenLimitExceeded(
            file_name=file_name,
            total_input_tokens=total_input_tokens,
            max_tokens=max_tokens,
        )
    return total_input_tokens


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def _resolve_client(client: OpenAI | None, *, api_key: str | None = None) -> OpenAI:
    if client is not None:
        return client

    if api_key:
        return OpenAI(api_key=api_key)

    _load_dotenv_if_available()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set")

    return OpenAI()


def _build_user_content(file_name: str, document_text: str) -> str:
    return f"File: {file_name}\n\nDocument text:\n{document_text}"


def _build_blank_output(schema: type[SchemaT]) -> SchemaT:
    return schema.model_construct(**{field_name: "" for field_name in schema.model_fields})


def _build_non_complete_result(
    *,
    pdf_path: str | Path,
    schema: type[SchemaT],
    model: str,
    status: str,
    input_tokens: int | None,
    error: str,
) -> ExtractionResult[SchemaT]:
    path = Path(pdf_path)
    return ExtractionResult(
        file_name=path.name,
        pdf_path=path,
        input_tokens=input_tokens,
        output=_build_blank_output(schema),
        output_tokens=None,
        model=model,
        status=status,
        skipped=(status == STATUS_SKIPPED),
        error=error,
    )


def _build_document_record(
    *,
    path: Path,
    input_tokens: int | None,
    custom_id: str | None,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "file_name": path.name,
        "pdf_path": str(path),
        "input_tokens": input_tokens,
        "custom_id": custom_id,
        "status": status,
        "error": error,
    }


def _count_docs_with_status(documents: Sequence[dict[str, Any]], status: str) -> int:
    return sum(1 for doc in documents if doc.get("status") == status)


def _build_doc_failed_result(
    *,
    doc: dict[str, Any],
    schema: type[SchemaT],
    model: str,
    error: str,
) -> ExtractionResult[SchemaT]:
    return _build_non_complete_result(
        pdf_path=doc["pdf_path"],
        schema=schema,
        model=model,
        status=STATUS_FAILED,
        input_tokens=doc.get("input_tokens"),
        error=error,
    )


def _result_from_doc_status(
    *,
    doc: dict[str, Any],
    schema: type[SchemaT],
    model: str,
) -> ExtractionResult[SchemaT] | None:
    doc_status = doc.get("status")
    if doc_status not in {STATUS_SKIPPED, STATUS_FAILED}:
        return None

    error_message = str(doc.get("error") or "")
    if doc_status == STATUS_FAILED and not error_message:
        error_message = "Failed before batch submission."

    return _build_non_complete_result(
        pdf_path=doc.get("pdf_path", ""),
        schema=schema,
        model=model,
        status=doc_status,
        input_tokens=doc.get("input_tokens"),
        error=error_message,
    )


def _extract_output_tokens(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    if isinstance(usage, dict):
        value = usage.get("output_tokens")
    else:
        value = getattr(usage, "output_tokens", None)

    return value if isinstance(value, int) else None


def _prepare_document(
    *,
    pdf_path: str | Path,
    prompt: str,
    max_input_tokens: int,
    count_tokens: Callable[[str], int],
) -> _PreparedDocument:
    path = Path(pdf_path)
    doc_text = extract_text_from_pdf(path)
    user_content = _build_user_content(path.name, doc_text)
    input_tokens = ensure_input_token_limit(
        base_prompt=prompt,
        user_content=user_content,
        max_tokens=max_input_tokens,
        file_name=path.name,
        count_tokens=count_tokens,
    )
    return _PreparedDocument(path=path, user_content=user_content, input_tokens=input_tokens)


def _extract_one(
    *,
    prepared: _PreparedDocument,
    schema: type[SchemaT],
    prompt: str,
    model: str,
    client: OpenAI,
) -> ExtractionResult[SchemaT]:
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": prepared.user_content},
        ],
        text_format=schema,
    )

    if response.output_parsed is None:
        raise ValueError(f"Model returned no parsed output for {prepared.path.name}")

    return ExtractionResult(
        file_name=prepared.path.name,
        pdf_path=prepared.path,
        input_tokens=prepared.input_tokens,
        output=schema.model_validate(response.output_parsed),
        output_tokens=_extract_output_tokens(response),
        model=model,
        status=STATUS_COMPLETE,
    )


def run_single_extraction(
    *,
    pdf_path: str | Path,
    schema: type[SchemaT],
    prompt: str,
    model: str,
    max_input_tokens: int,
    client: OpenAI,
    count_tokens: Callable[[str], int] | None = None,
) -> ExtractionResult[SchemaT]:
    """Run extraction for one PDF."""

    _ensure_positive_max_input_tokens(max_input_tokens)

    token_counter = count_tokens or make_token_counter(model)
    prepared = _prepare_document(
        pdf_path=pdf_path,
        prompt=prompt,
        max_input_tokens=max_input_tokens,
        count_tokens=token_counter,
    )
    return _extract_one(
        prepared=prepared,
        schema=schema,
        prompt=prompt,
        model=model,
        client=client,
    )


def run_directory_extraction(
    *,
    config: ExtractAIConfig,
    schema: type[SchemaT],
    prompt: str,
    client: OpenAI | None = None,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> list[ExtractionResult[SchemaT]]:
    """Run synchronous extraction over every PDF in config.pdf_dir."""

    _ensure_positive_max_input_tokens(config.max_input_tokens)

    resolved_client = _resolve_client(client, api_key=config.api_key)
    token_counter = make_token_counter(config.model)
    pdf_files = list_pdf_files(config.pdf_dir)
    total_files = len(pdf_files)

    if progress_callback is not None:
        progress_callback(0, total_files, Path(""))

    results: list[ExtractionResult[SchemaT]] = []
    for processed_count, pdf_path in enumerate(pdf_files, start=1):
        try:
            prepared = _prepare_document(
                pdf_path=pdf_path,
                prompt=prompt,
                max_input_tokens=config.max_input_tokens,
                count_tokens=token_counter,
            )
            results.append(
                _extract_one(
                    prepared=prepared,
                    schema=schema,
                    prompt=prompt,
                    model=config.model,
                    client=resolved_client,
                )
            )
        except InputTokenLimitExceeded as exc:
            print(
                f"Skipping {exc.file_name}: {exc.total_input_tokens} tokens exceeds "
                f"the {exc.max_tokens} token limit."
            )
            results.append(
                _build_non_complete_result(
                    pdf_path=pdf_path,
                    schema=schema,
                    model=config.model,
                    status=STATUS_SKIPPED,
                    input_tokens=exc.total_input_tokens,
                    error=str(exc),
                )
            )
        except Exception as exc:
            file_name = Path(pdf_path).name
            print(f"Failed {file_name}: {exc}")
            results.append(
                _build_non_complete_result(
                    pdf_path=pdf_path,
                    schema=schema,
                    model=config.model,
                    status=STATUS_FAILED,
                    input_tokens=None,
                    error=str(exc),
                )
            )
        finally:
            if progress_callback is not None:
                progress_callback(processed_count, total_files, Path(pdf_path))

    return results


def _batch_store_dir(output_dir: str | Path | None) -> Path:
    base = Path(output_dir) if output_dir is not None else Path.cwd()
    return base / ".extractai" / "batches"


def _submission_dir(*, submission_id: str, output_dir: str | Path | None = None) -> Path:
    return _batch_store_dir(output_dir) / submission_id


def _manifest_path(*, submission_id: str, output_dir: str | Path | None = None) -> Path:
    return _submission_dir(submission_id=submission_id, output_dir=output_dir) / "manifest.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for raw_line in text.splitlines() if (line := raw_line.strip())]


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    for serializer_name in ("model_dump", "to_dict"):
        serializer = getattr(value, serializer_name, None)
        if callable(serializer):
            dumped = serializer()
            if isinstance(dumped, dict):
                return dumped
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": str(value)}


def _file_content_to_text(file_response: Any) -> str:
    text_attr = getattr(file_response, "text", None)
    if callable(text_attr):
        value = text_attr()
        if isinstance(value, str):
            return value
    if isinstance(text_attr, str):
        return text_attr

    content_attr = getattr(file_response, "content", None)
    if isinstance(content_attr, bytes):
        return content_attr.decode("utf-8")
    if isinstance(content_attr, str):
        return content_attr

    return str(file_response)


def _download_jsonl_lines(*, client: OpenAI, file_id: str, output_path: Path) -> list[dict[str, Any]]:
    text = _file_content_to_text(client.files.content(file_id))
    output_path.write_text(text, encoding="utf-8")
    return _parse_jsonl_text(text)


def _to_unix_seconds(created_after: date | datetime | str | int | float | None) -> int | None:
    if created_after is None:
        return None

    if isinstance(created_after, (int, float)):
        return int(created_after)

    if isinstance(created_after, datetime):
        dt = created_after if created_after.tzinfo else created_after.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    if isinstance(created_after, date):
        dt = datetime.combine(created_after, datetime.min.time(), tzinfo=timezone.utc)
        return int(dt.timestamp())

    if isinstance(created_after, str):
        text = created_after.strip()
        if not text:
            raise ValueError("created_after cannot be an empty string")

        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed_date = date.fromisoformat(text)
            except ValueError as exc:
                raise ValueError(
                    "created_after must be ISO date/datetime, unix timestamp, date, or datetime"
                ) from exc
            dt = datetime.combine(parsed_date, datetime.min.time(), tzinfo=timezone.utc)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    raise ValueError("created_after must be ISO date/datetime, unix timestamp, date, or datetime")


def list_openai_models(
    *,
    client: OpenAI | None = None,
    api_key: str | None = None,
    created_after: date | datetime | str | int | float | None = None,
) -> list[str]:
    """List model IDs available to the current OpenAI account."""

    resolved_client = _resolve_client(client, api_key=api_key)
    response = resolved_client.models.list()
    created_after_ts = _to_unix_seconds(created_after)

    data = getattr(response, "data", None)
    if not isinstance(data, list):
        response_dict = _to_dict(response)
        raw_data = response_dict.get("data")
        data = raw_data if isinstance(raw_data, list) else []

    model_ids: list[str] = []
    for item in data:
        model_data = _to_dict(item)
        model_id = model_data.get("id")
        if not isinstance(model_id, str):
            continue

        if created_after_ts is not None:
            created_value = model_data.get("created")
            if not isinstance(created_value, int):
                continue
            if created_value <= created_after_ts:
                continue

        model_ids.append(model_id)

    return sorted(model_ids)


def _build_batch_task(*, custom_id: str, model: str, prompt: str, user_content: str) -> dict[str, Any]:
    # Chat Completions with response_format=json_object requires the messages
    # to explicitly mention JSON output.
    batch_prompt = (
        f"{prompt.rstrip()}\n\n"
        "Return output as a valid json object only."
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": batch_prompt},
                {"role": "user", "content": user_content},
            ],
        },
    }


def _extract_chat_completion_content(body: dict[str, Any]) -> Any:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response body missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("response choice is not an object")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("response choice missing message")

    content = message.get("content")
    if isinstance(content, list):
        text_parts = [
            item.get("text")
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "output_text"}
            and isinstance(item.get("text"), str)
        ]
        if not text_parts:
            raise ValueError("response message content list has no text blocks")
        return "".join(text_parts)

    return content


def _load_manifest(*, submission_id: str, output_dir: str | Path | None = None) -> dict[str, Any]:
    path = _manifest_path(submission_id=submission_id, output_dir=output_dir)
    if not path.exists():
        raise FileNotFoundError(f"No manifest found for submission '{submission_id}' at {path}")
    return _read_json(path)


def submit_directory_batch(
    *,
    config: ExtractAIConfig,
    schema: type[SchemaT],
    prompt: str,
    output_dir: str | Path | None = None,
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
    client: OpenAI | None = None,
) -> BatchSubmission:
    """Create and submit a batch for all processable PDFs in a directory."""

    _ensure_positive_max_input_tokens(config.max_input_tokens)
    if completion_window != "24h":
        raise ValueError("completion_window must be '24h'")

    validated_prompt = build_prompt(prompt)
    token_counter = make_token_counter(config.model)

    store_dir = _batch_store_dir(output_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    working_id = f"submission_{int(time.time())}_{uuid4().hex[:8]}"
    working_dir = store_dir / working_id
    working_dir.mkdir(parents=True, exist_ok=False)

    input_jsonl_path = working_dir / "batch_input.jsonl"
    tasks: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []

    for index, pdf_path in enumerate(list_pdf_files(config.pdf_dir)):
        path = Path(pdf_path)

        try:
            prepared = _prepare_document(
                pdf_path=path,
                prompt=validated_prompt,
                max_input_tokens=config.max_input_tokens,
                count_tokens=token_counter,
            )
            custom_id = f"doc-{index}"
            tasks.append(
                _build_batch_task(
                    custom_id=custom_id,
                    model=config.model,
                    prompt=validated_prompt,
                    user_content=prepared.user_content,
                )
            )
            documents.append(
                _build_document_record(
                    path=path,
                    input_tokens=prepared.input_tokens,
                    custom_id=custom_id,
                    status="PENDING",
                    error=None,
                )
            )
        except InputTokenLimitExceeded as exc:
            print(
                f"Skipping {exc.file_name}: {exc.total_input_tokens} tokens exceeds "
                f"the {exc.max_tokens} token limit."
            )
            documents.append(
                _build_document_record(
                    path=path,
                    input_tokens=exc.total_input_tokens,
                    custom_id=None,
                    status=STATUS_SKIPPED,
                    error=str(exc),
                )
            )
        except Exception as exc:
            print(f"Failed {path.name}: {exc}")
            documents.append(
                _build_document_record(
                    path=path,
                    input_tokens=None,
                    custom_id=None,
                    status=STATUS_FAILED,
                    error=str(exc),
                )
            )

    _write_jsonl(input_jsonl_path, tasks)

    skipped_requests = _count_docs_with_status(documents, STATUS_SKIPPED)
    failed_requests = _count_docs_with_status(documents, STATUS_FAILED)

    if not tasks:
        submission_id = f"local_{int(time.time())}_{uuid4().hex[:8]}"
        final_dir = store_dir / submission_id
        working_dir.rename(final_dir)

        manifest = {
            "version": 1,
            "submission_id": submission_id,
            "batch_id": None,
            "input_file_id": None,
            "local_only": True,
            "model": config.model,
            "schema_fields": list(schema.model_fields.keys()),
            "prompt": validated_prompt,
            "created_at": int(time.time()),
            "documents": documents,
        }
        manifest_path = final_dir / "manifest.json"
        _write_json(manifest_path, manifest)

        return BatchSubmission(
            submission_id=submission_id,
            batch_id=None,
            input_file_id=None,
            batch_dir=final_dir,
            manifest_path=manifest_path,
            input_jsonl_path=final_dir / "batch_input.jsonl",
            submitted_requests=0,
            skipped_requests=skipped_requests,
            failed_requests=failed_requests,
            status="LOCAL_ONLY",
        )

    resolved_client = _resolve_client(client, api_key=config.api_key)
    with input_jsonl_path.open("rb") as handle:
        input_file = resolved_client.files.create(file=handle, purpose="batch")

    batch = resolved_client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata=metadata,
    )

    batch_id = getattr(batch, "id", None)
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("Batch creation did not return a valid batch id")

    final_dir = store_dir / batch_id
    if final_dir.exists():
        raise FileExistsError(f"Batch storage directory already exists: {final_dir}")
    working_dir.rename(final_dir)

    manifest = {
        "version": 1,
        "submission_id": batch_id,
        "batch_id": batch_id,
        "input_file_id": input_file.id,
        "local_only": False,
        "model": config.model,
        "schema_fields": list(schema.model_fields.keys()),
        "prompt": validated_prompt,
        "created_at": int(time.time()),
        "documents": documents,
    }
    manifest_path = final_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    return BatchSubmission(
        submission_id=batch_id,
        batch_id=batch_id,
        input_file_id=input_file.id,
        batch_dir=final_dir,
        manifest_path=manifest_path,
        input_jsonl_path=final_dir / "batch_input.jsonl",
        submitted_requests=len(tasks),
        skipped_requests=skipped_requests,
        failed_requests=failed_requests,
        status="SUBMITTED",
    )


def get_batch_status(
    *,
    submission_id: str,
    output_dir: str | Path | None = None,
    client: OpenAI | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Return current status details for a batch submission."""

    manifest = _load_manifest(submission_id=submission_id, output_dir=output_dir)
    documents = manifest.get("documents", [])
    if not isinstance(documents, list):
        documents = []

    if manifest.get("local_only"):
        normalized_docs = [doc for doc in documents if isinstance(doc, dict)]
        failed_count = (
            _count_docs_with_status(normalized_docs, STATUS_FAILED)
            + _count_docs_with_status(normalized_docs, STATUS_SKIPPED)
        )
        return {
            "id": submission_id,
            "status": "completed",
            "local_only": True,
            "request_counts": {
                "total": len(documents),
                "completed": 0,
                "failed": failed_count,
            },
        }

    resolved_client = _resolve_client(client, api_key=api_key)
    batch = resolved_client.batches.retrieve(manifest["batch_id"])
    return _to_dict(batch)


def _parse_chat_completion_line(
    *,
    line: dict[str, Any],
    schema: type[SchemaT],
    doc: dict[str, Any],
    model: str,
) -> ExtractionResult[SchemaT]:
    custom_id = line.get("custom_id")

    def fail(message: str) -> ExtractionResult[SchemaT]:
        return _build_doc_failed_result(
            doc=doc,
            schema=schema,
            model=model,
            error=message,
        )

    if line.get("error") is not None:
        return fail(f"Batch request {custom_id} failed: {line['error']}")

    response = line.get("response")
    if not isinstance(response, dict):
        return fail(f"Batch request {custom_id} has no response payload.")

    if response.get("status_code") != 200:
        return fail(f"Batch request {custom_id} returned status {response.get('status_code')}")

    body = response.get("body")
    if not isinstance(body, dict):
        return fail(f"Batch request {custom_id} missing response body.")

    try:
        content = _extract_chat_completion_content(body)
        payload = json.loads(content) if isinstance(content, str) else content
        output = schema.model_validate(payload)
    except Exception as exc:
        return fail(f"Batch request {custom_id} could not be parsed: {exc}")

    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    output_tokens = usage.get("completion_tokens")
    if not isinstance(output_tokens, int):
        output_tokens = usage.get("output_tokens") if isinstance(usage.get("output_tokens"), int) else None

    response_model = body.get("model") if isinstance(body.get("model"), str) else model

    return ExtractionResult(
        file_name=doc["file_name"],
        pdf_path=Path(doc["pdf_path"]),
        input_tokens=doc.get("input_tokens"),
        output=output,
        output_tokens=output_tokens,
        model=response_model,
        status=STATUS_COMPLETE,
    )


def collect_batch_results(
    *,
    submission_id: str,
    schema: type[SchemaT],
    output_dir: str | Path | None = None,
    client: OpenAI | None = None,
    api_key: str | None = None,
    csv_file_name: str = "extractai_results.csv",
) -> tuple[list[ExtractionResult[SchemaT]], Path]:
    """Collect batch output, map back to documents, and write CSV."""

    manifest = _load_manifest(submission_id=submission_id, output_dir=output_dir)
    documents = manifest.get("documents", [])
    model = manifest.get("model")
    submission_dir = _submission_dir(submission_id=submission_id, output_dir=output_dir)

    if not isinstance(documents, list):
        raise ValueError("Invalid batch manifest: documents must be a list")
    if not isinstance(model, str):
        raise ValueError("Invalid batch manifest: model is missing")

    by_custom_id: dict[str, dict[str, Any]] = {
        doc["custom_id"]: doc
        for doc in documents
        if isinstance(doc, dict) and isinstance(doc.get("custom_id"), str)
    }

    parsed_results: dict[str, ExtractionResult[SchemaT]] = {}

    if not manifest.get("local_only"):
        resolved_client = _resolve_client(client, api_key=api_key)
        batch = resolved_client.batches.retrieve(manifest["batch_id"])
        batch_data = _to_dict(batch)
        _write_json(submission_dir / BATCH_STATUS_FILE_NAME, batch_data)

        status = batch_data.get("status")
        if status not in TERMINAL_BATCH_STATUSES:
            raise RuntimeError(
                f"Batch {manifest['batch_id']} is not complete yet. Current status: {status}"
            )

        output_file_id = batch_data.get("output_file_id")
        if isinstance(output_file_id, str) and output_file_id:
            for line in _download_jsonl_lines(
                client=resolved_client,
                file_id=output_file_id,
                output_path=submission_dir / BATCH_OUTPUT_FILE_NAME,
            ):
                custom_id = line.get("custom_id")
                if not isinstance(custom_id, str):
                    continue
                doc = by_custom_id.get(custom_id)
                if doc is None:
                    continue
                parsed_results[custom_id] = _parse_chat_completion_line(
                    line=line,
                    schema=schema,
                    doc=doc,
                    model=model,
                )

        error_file_id = batch_data.get("error_file_id")
        if isinstance(error_file_id, str) and error_file_id:
            for line in _download_jsonl_lines(
                client=resolved_client,
                file_id=error_file_id,
                output_path=submission_dir / BATCH_ERROR_FILE_NAME,
            ):
                custom_id = line.get("custom_id")
                if not isinstance(custom_id, str) or custom_id in parsed_results:
                    continue
                doc = by_custom_id.get(custom_id)
                if doc is None:
                    continue
                parsed_results[custom_id] = _build_doc_failed_result(
                    doc=doc,
                    schema=schema,
                    model=model,
                    error=f"Batch request {custom_id} failed: {line.get('error')}",
                )

    final_results: list[ExtractionResult[SchemaT]] = []

    for doc in documents:
        if not isinstance(doc, dict):
            continue

        non_complete_result = _result_from_doc_status(
            doc=doc,
            schema=schema,
            model=model,
        )
        if non_complete_result is not None:
            final_results.append(non_complete_result)
            continue

        custom_id = doc.get("custom_id")
        if isinstance(custom_id, str) and custom_id in parsed_results:
            final_results.append(parsed_results[custom_id])
            continue

        final_results.append(
            _build_doc_failed_result(
                doc=doc,
                schema=schema,
                model=model,
                error="No batch result returned for request.",
            )
        )

    csv_path = save_results_to_csv(final_results, output_dir=output_dir, file_name=csv_file_name)
    return final_results, csv_path


def run_extraction(
    *,
    config: ExtractAIConfig,
    schema: type[SchemaT],
    prompt: str,
    client: OpenAI | None = None,
    output_dir: str | Path | None = None,
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
) -> list[ExtractionResult[SchemaT]] | BatchSubmission:
    """Entry point for sync or batch mode based on config.use_batch."""

    if config.use_batch:
        return submit_directory_batch(
            config=config,
            schema=schema,
            prompt=prompt,
            output_dir=output_dir,
            completion_window=completion_window,
            metadata=metadata,
            client=client,
        )

    return run_directory_extraction(
        config=config,
        schema=schema,
        prompt=prompt,
        client=client,
    )


def _collect_output_fields(results: Sequence[ExtractionResult[SchemaT]]) -> list[str]:
    return list(dict.fromkeys(key for result in results for key in result.output.model_dump().keys()))


def _result_to_csv_row(result: ExtractionResult[SchemaT]) -> dict[str, Any]:
    return {
        "id": result.file_name,
        **result.output.model_dump(),
        "model": result.model,
        "status": result.status,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


def save_results_to_csv(
    results: Sequence[ExtractionResult[SchemaT]],
    *,
    output_dir: str | Path | None = None,
    file_name: str = "extractai_results.csv",
) -> Path:
    """Write extraction results to CSV and return its path."""

    target_dir = Path(output_dir) if output_dir is not None else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / file_name

    output_fields = _collect_output_fields(results)
    fieldnames = ["id", *output_fields, "model", "status", "input_tokens", "output_tokens"]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(_result_to_csv_row(result))

    return csv_path


__all__ = [
    "BatchSubmission",
    "DEFAULT_MAX_INPUT_TOKENS",
    "ExtractAIConfig",
    "ExtractionResult",
    "InputTokenLimitExceeded",
    "build_prompt",
    "collect_batch_results",
    "ensure_input_token_limit",
    "extract_text_from_pdf",
    "get_batch_status",
    "list_pdf_files",
    "list_openai_models",
    "make_token_counter",
    "run_extraction",
    "run_directory_extraction",
    "run_single_extraction",
    "save_results_to_csv",
    "submit_directory_batch",
]
