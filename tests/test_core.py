from __future__ import annotations

import builtins
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

from pydantic import BaseModel, ValidationError

import extractai
import extractai.core as core


class ExtractedDocument(BaseModel):
    summary: str
    category: Literal["Financial", "Research", "Government", "Other"]


class _FakePage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class _FakePDF:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> "_FakePDF":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeResponses:
    def __init__(self, parsed_outputs: object | list[object], output_tokens: int | None | list[int | None] = None) -> None:
        self._parsed_outputs = parsed_outputs if isinstance(parsed_outputs, list) else [parsed_outputs]
        self._output_tokens = output_tokens if isinstance(output_tokens, list) else [output_tokens] * len(self._parsed_outputs)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self._parsed_outputs:
            raise AssertionError("No more fake outputs configured")

        parsed_output = self._parsed_outputs.pop(0)
        if isinstance(parsed_output, Exception):
            raise parsed_output

        output_token_value = self._output_tokens.pop(0) if self._output_tokens else None
        usage = SimpleNamespace(output_tokens=output_token_value) if output_token_value is not None else None
        return SimpleNamespace(output_parsed=parsed_output, usage=usage)


class _FakeClient:
    def __init__(self, parsed_outputs: object | list[object], output_tokens: int | None | list[int | None] = None) -> None:
        self.responses = _FakeResponses(parsed_outputs, output_tokens=output_tokens)


class _FakeBatchClient:
    def __init__(self) -> None:
        self._output_by_file_id: dict[str, str] = {}
        self.files = SimpleNamespace(create=self._files_create, content=self._files_content)
        self.batches = SimpleNamespace(create=self._batches_create, retrieve=self._batches_retrieve)

    def _files_create(self, **kwargs):
        _ = kwargs
        return SimpleNamespace(id="file_batch_input_1")

    def _batches_create(self, **kwargs):
        _ = kwargs
        return SimpleNamespace(id="batch_123")

    def _batches_retrieve(self, batch_id: str):
        _ = batch_id
        return SimpleNamespace(
            id="batch_123",
            status="completed",
            output_file_id="file_output_1",
            error_file_id="file_error_1",
        )

    def _files_content(self, file_id: str):
        return SimpleNamespace(text=self._output_by_file_id.get(file_id, ""))


class _FakeModelsClient:
    def __init__(self, models: list[dict[str, object]]) -> None:
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=models))


class BuildPromptTests(unittest.TestCase):
    def test_build_prompt_returns_user_defined_prompt(self) -> None:
        prompt_text = """
        Extract fields from the text.
        Category must be one of Financial, Research, Government, Other.
        """
        self.assertEqual(core.build_prompt(prompt_text), prompt_text.strip())

    def test_build_prompt_requires_non_empty_prompt(self) -> None:
        with self.assertRaises(ValueError):
            core.build_prompt("   ")


class PackageImportTests(unittest.TestCase):
    def test_extractai_exports_main_api(self) -> None:
        self.assertTrue(hasattr(extractai, "run_directory_extraction"))
        self.assertTrue(hasattr(extractai, "submit_directory_batch"))


class ConfigAndTokenTests(unittest.TestCase):
    def test_config_default_max_tokens(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data")
        self.assertEqual(cfg.max_input_tokens, 50_000)

    def test_ensure_input_token_limit_allows_under_limit(self) -> None:
        total = core.ensure_input_token_limit(
            base_prompt="abc",
            user_content="defg",
            max_tokens=10,
            file_name="doc.pdf",
            count_tokens=len,
        )
        self.assertEqual(total, 7)

    def test_ensure_input_token_limit_raises_at_limit(self) -> None:
        with self.assertRaises(ValueError):
            core.ensure_input_token_limit(
                base_prompt="abc",
                user_content="defg",
                max_tokens=7,
                file_name="doc.pdf",
                count_tokens=len,
            )

    def test_make_token_counter_fallback_when_tiktoken_missing(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            counter = core.make_token_counter("any-model")

        self.assertEqual(counter("abcd"), 1)
        self.assertEqual(counter("abcdefgh"), 2)


class ModelListingTests(unittest.TestCase):
    def test_list_openai_models_returns_sorted_model_ids(self) -> None:
        fake_client = _FakeModelsClient(
            [
                {"id": "gpt-5", "object": "model", "created": 2, "owned_by": "openai"},
                {"id": "gpt-4.1", "object": "model", "created": 1, "owned_by": "openai"},
            ]
        )

        models = core.list_openai_models(client=fake_client)  # type: ignore[arg-type]

        self.assertEqual(models, ["gpt-4.1", "gpt-5"])

    def test_list_openai_models_uses_api_key_when_client_not_supplied(self) -> None:
        fake_client = _FakeModelsClient([{"id": "gpt-5", "object": "model"}])

        with patch("extractai.core.OpenAI", return_value=fake_client) as mock_openai:
            models = core.list_openai_models(api_key="direct-api-key")

        mock_openai.assert_called_once_with(api_key="direct-api-key")
        self.assertEqual(models[0], "gpt-5")

    def test_list_openai_models_filters_by_created_after_date_string(self) -> None:
        fake_client = _FakeModelsClient(
            [
                {"id": "gpt-old", "created": 1735689600},  # 2025-01-01T00:00:00Z
                {"id": "gpt-new", "created": 1767225600},  # 2026-01-01T00:00:00Z
            ]
        )

        models = core.list_openai_models(
            client=fake_client,  # type: ignore[arg-type]
            created_after="2025-06-01",
        )

        self.assertEqual(models, ["gpt-new"])


class FileTests(unittest.TestCase):
    def test_list_pdf_files_returns_sorted_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "b.pdf").write_text("x", encoding="utf-8")
            (tmp_path / "a.pdf").write_text("x", encoding="utf-8")
            (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
            files = core.list_pdf_files(tmp_path)
            self.assertEqual([f.name for f in files], ["a.pdf", "b.pdf"])

    def test_list_pdf_files_raises_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                core.list_pdf_files(tmp)

    def test_extract_text_from_pdf_joins_pages(self) -> None:
        fake_pdf = _FakePDF([_FakePage("Page one"), _FakePage(None), _FakePage("Page three")])
        with patch("extractai.core.pdfplumber.open", return_value=fake_pdf):
            text = core.extract_text_from_pdf("dummy.pdf")
        self.assertEqual(text, "Page one\n\nPage three")


class SingleExtractionTests(unittest.TestCase):
    def test_run_single_extraction_requires_positive_max_tokens(self) -> None:
        client = _FakeClient({"summary": "short", "category": "Other"})
        with self.assertRaises(ValueError):
            core.run_single_extraction(
                pdf_path="doc.pdf",
                schema=ExtractedDocument,
                prompt="prompt",
                model="gpt-5-nano",
                max_input_tokens=0,
                client=client,
            )

    def test_run_single_extraction_returns_structured_result(self) -> None:
        client = _FakeClient({"summary": "short", "category": "Other"}, output_tokens=77)
        with patch("extractai.core.extract_text_from_pdf", return_value="hello world"):
            result = core.run_single_extraction(
                pdf_path="doc.pdf",
                schema=ExtractedDocument,
                prompt="prompt",
                model="gpt-5-nano",
                max_input_tokens=500,
                client=client,
                count_tokens=len,
            )

        self.assertEqual(result.file_name, "doc.pdf")
        self.assertEqual(result.output.summary, "short")
        self.assertEqual(result.output.category, "Other")
        self.assertGreater(result.input_tokens, 0)
        self.assertEqual(result.output_tokens, 77)
        self.assertEqual(result.model, "gpt-5-nano")
        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(client.responses.calls[0]["model"], "gpt-5-nano")

    def test_run_single_extraction_uses_default_token_counter_when_not_provided(self) -> None:
        client = _FakeClient({"summary": "short", "category": "Other"})
        with patch("extractai.core.extract_text_from_pdf", return_value="hello world"), patch(
            "extractai.core.make_token_counter", return_value=len
        ) as mock_make_counter:
            result = core.run_single_extraction(
                pdf_path="doc.pdf",
                schema=ExtractedDocument,
                prompt="prompt",
                model="gpt-5-nano",
                max_input_tokens=500,
                client=client,
            )

        self.assertEqual(result.output.summary, "short")
        mock_make_counter.assert_called_once_with("gpt-5-nano")

    def test_run_single_extraction_reads_output_tokens_from_dict_usage(self) -> None:
        class _ClientWithDictUsage:
            class _Responses:
                @staticmethod
                def parse(**kwargs):
                    return SimpleNamespace(
                        output_parsed={"summary": "short", "category": "Other"},
                        usage={"output_tokens": 88},
                    )

            responses = _Responses()

        with patch("extractai.core.extract_text_from_pdf", return_value="hello world"):
            result = core.run_single_extraction(
                pdf_path="doc.pdf",
                schema=ExtractedDocument,
                prompt="prompt",
                model="gpt-5-nano",
                max_input_tokens=500,
                client=_ClientWithDictUsage(),  # type: ignore[arg-type]
                count_tokens=len,
            )

        self.assertEqual(result.output_tokens, 88)

    def test_run_single_extraction_raises_on_token_limit(self) -> None:
        client = _FakeClient({"summary": "short", "category": "Other"})
        with patch("extractai.core.extract_text_from_pdf", return_value="hello world"):
            with self.assertRaises(ValueError):
                core.run_single_extraction(
                    pdf_path="doc.pdf",
                    schema=ExtractedDocument,
                    prompt="prompt",
                    model="gpt-5-nano",
                    max_input_tokens=1,
                    client=client,
                    count_tokens=len,
                )

    def test_run_single_extraction_raises_when_model_returns_no_output(self) -> None:
        client = _FakeClient(None)
        with patch("extractai.core.extract_text_from_pdf", return_value="hello world"):
            with self.assertRaises(ValueError):
                core.run_single_extraction(
                    pdf_path="doc.pdf",
                    schema=ExtractedDocument,
                    prompt="prompt",
                    model="gpt-5-nano",
                    max_input_tokens=500,
                    client=client,
                    count_tokens=len,
                )

    def test_run_single_extraction_raises_on_invalid_schema_output(self) -> None:
        client = _FakeClient({"summary": "short", "category": "InvalidCategory"})
        with patch("extractai.core.extract_text_from_pdf", return_value="hello world"):
            with self.assertRaises(ValidationError):
                core.run_single_extraction(
                    pdf_path="doc.pdf",
                    schema=ExtractedDocument,
                    prompt="prompt",
                    model="gpt-5-nano",
                    max_input_tokens=500,
                    client=client,
                    count_tokens=len,
                )


class DirectoryExtractionTests(unittest.TestCase):
    def test_run_directory_extraction_requires_positive_max_tokens(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", max_input_tokens=0)
        with self.assertRaises(ValueError):
            core.run_directory_extraction(
                config=cfg,
                schema=ExtractedDocument,
                prompt="prompt",
                client=_FakeClient({"summary": "s", "category": "Other"}),
            )

    def test_run_directory_extraction_with_supplied_client_processes_all_pdfs(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", model="gpt-5-nano", max_input_tokens=500)
        fake_client = _FakeClient([
            {"summary": "sum a", "category": "Other"},
            {"summary": "sum b", "category": "Financial"},
        ])

        with patch("extractai.core.list_pdf_files", return_value=[Path("a.pdf"), Path("b.pdf")]), patch(
            "extractai.core.extract_text_from_pdf", side_effect=["text a", "text b"]
        ), patch("extractai.core.make_token_counter", return_value=len):
            results = core.run_directory_extraction(
                config=cfg,
                schema=ExtractedDocument,
                prompt="prompt",
                client=fake_client,
            )

        self.assertEqual([r.file_name for r in results], ["a.pdf", "b.pdf"])
        self.assertEqual(results[0].output.category, "Other")
        self.assertEqual(results[1].output.category, "Financial")
        self.assertEqual(results[0].status, "COMPLETE")
        self.assertEqual(results[1].status, "COMPLETE")
        self.assertEqual(results[0].model, "gpt-5-nano")
        self.assertEqual(len(fake_client.responses.calls), 2)

    def test_run_directory_extraction_without_client_requires_api_key(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data")
        with patch("extractai.core._load_dotenv_if_available"), patch(
            "extractai.core.list_pdf_files", return_value=[Path("a.pdf")]
        ), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvironmentError):
                core.run_directory_extraction(
                    config=cfg,
                    schema=ExtractedDocument,
                    prompt="prompt",
                )

    def test_run_directory_extraction_without_client_uses_openai_client(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", max_input_tokens=500)
        fake_client = _FakeClient({"summary": "s", "category": "Other"})

        with patch("extractai.core.list_pdf_files", return_value=[Path("a.pdf")]), patch(
            "extractai.core.extract_text_from_pdf", return_value="text a"
        ), patch("extractai.core.make_token_counter", return_value=len), patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True
        ), patch("extractai.core.OpenAI", return_value=fake_client) as mock_openai:
            results = core.run_directory_extraction(
                config=cfg,
                schema=ExtractedDocument,
                prompt="prompt",
            )

        self.assertEqual(len(results), 1)
        mock_openai.assert_called_once_with()
        self.assertEqual(results[0].status, "COMPLETE")
        self.assertEqual(results[0].model, "gpt-5-nano")

    def test_run_directory_extraction_without_client_uses_config_api_key(self) -> None:
        cfg = core.ExtractAIConfig(
            pdf_dir="sample_data",
            max_input_tokens=500,
            api_key="test-key-from-config",
        )
        fake_client = _FakeClient({"summary": "s", "category": "Other"})

        with patch("extractai.core.list_pdf_files", return_value=[Path("a.pdf")]), patch(
            "extractai.core.extract_text_from_pdf", return_value="text a"
        ), patch("extractai.core.make_token_counter", return_value=len), patch.dict(
            os.environ, {}, clear=True
        ), patch("extractai.core.OpenAI", return_value=fake_client) as mock_openai:
            results = core.run_directory_extraction(
                config=cfg,
                schema=ExtractedDocument,
                prompt="prompt",
            )

        self.assertEqual(len(results), 1)
        mock_openai.assert_called_once_with(api_key="test-key-from-config")
        self.assertEqual(results[0].status, "COMPLETE")

    def test_run_directory_extraction_skips_over_limit_files_and_returns_blank_row(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", model="gpt-5-nano", max_input_tokens=200)
        fake_client = _FakeClient({"summary": "kept", "category": "Other"})

        with patch("extractai.core.list_pdf_files", return_value=[Path("too_big.pdf"), Path("ok.pdf")]), patch(
            "extractai.core.extract_text_from_pdf", side_effect=["x" * 1000, "short text"]
        ), patch("extractai.core.make_token_counter", return_value=len), patch("builtins.print") as mock_print:
            results = core.run_directory_extraction(
                config=cfg,
                schema=ExtractedDocument,
                prompt="prompt",
                client=fake_client,
            )

        self.assertEqual(len(results), 2)

        skipped = results[0]
        self.assertEqual(skipped.file_name, "too_big.pdf")
        self.assertTrue(skipped.skipped)
        self.assertEqual(skipped.status, "SKIPPED")
        self.assertEqual(skipped.model, "gpt-5-nano")
        self.assertEqual(skipped.output.summary, "")
        self.assertEqual(skipped.output.category, "")
        self.assertIsNone(skipped.output_tokens)
        self.assertIsNotNone(skipped.error)

        kept = results[1]
        self.assertFalse(kept.skipped)
        self.assertEqual(kept.status, "COMPLETE")
        self.assertEqual(kept.output.summary, "kept")
        self.assertEqual(kept.output.category, "Other")

        self.assertTrue(mock_print.called)
        printed = " ".join(str(arg) for arg in mock_print.call_args_list[0].args)
        self.assertIn("Skipping too_big.pdf", printed)
        self.assertIn("exceeds the 200 token limit", printed)

    def test_run_directory_extraction_marks_failed_and_continues(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", model="gpt-5-nano", max_input_tokens=500)
        fake_client = _FakeClient([
            RuntimeError("network lost"),
            {"summary": "ok", "category": "Other"},
        ])

        with patch("extractai.core.list_pdf_files", return_value=[Path("a.pdf"), Path("b.pdf")]), patch(
            "extractai.core.extract_text_from_pdf", side_effect=["text a", "text b"]
        ), patch("extractai.core.make_token_counter", return_value=len), patch("builtins.print") as mock_print:
            results = core.run_directory_extraction(
                config=cfg,
                schema=ExtractedDocument,
                prompt="prompt",
                client=fake_client,
            )

        self.assertEqual(len(results), 2)

        failed = results[0]
        self.assertEqual(failed.file_name, "a.pdf")
        self.assertEqual(failed.status, "FAILED")
        self.assertEqual(failed.model, "gpt-5-nano")
        self.assertIsNone(failed.input_tokens)
        self.assertEqual(failed.output.summary, "")
        self.assertEqual(failed.output.category, "")
        self.assertIsNotNone(failed.error)

        complete = results[1]
        self.assertEqual(complete.file_name, "b.pdf")
        self.assertEqual(complete.status, "COMPLETE")
        self.assertEqual(complete.output.summary, "ok")

        self.assertTrue(mock_print.called)
        printed = " ".join(str(arg) for arg in mock_print.call_args_list[0].args)
        self.assertIn("Failed a.pdf", printed)


class CsvExportTests(unittest.TestCase):
    def test_save_results_to_csv_with_output_dir(self) -> None:
        results = [
            core.ExtractionResult(
                file_name="doc_a.pdf",
                pdf_path=Path("doc_a.pdf"),
                input_tokens=11,
                output=ExtractedDocument(summary="summary a", category="Other"),
                output_tokens=101,
                model="gpt-5-nano",
                status="COMPLETE",
            ),
            core.ExtractionResult(
                file_name="doc_b.pdf",
                pdf_path=Path("doc_b.pdf"),
                input_tokens=22,
                output=ExtractedDocument(summary="summary b", category="Financial"),
                output_tokens=202,
                model="gpt-5-nano",
                status="SKIPPED",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_path = core.save_results_to_csv(results, output_dir=tmp, file_name="results.csv")

            self.assertEqual(out_path, Path(tmp) / "results.csv")
            self.assertTrue(out_path.exists())

            with out_path.open("r", encoding="utf-8", newline="") as handle:
                header = handle.readline().strip()
                self.assertEqual(header, "id,summary,category,model,status,input_tokens,output_tokens")
                handle.seek(0)
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["id"], "doc_a.pdf")
            self.assertEqual(rows[1]["id"], "doc_b.pdf")
            self.assertEqual(rows[0]["category"], "Other")
            self.assertEqual(rows[1]["category"], "Financial")
            self.assertEqual(rows[0]["model"], "gpt-5-nano")
            self.assertEqual(rows[1]["status"], "SKIPPED")
            self.assertEqual(rows[0]["output_tokens"], "101")
            self.assertEqual(rows[1]["output_tokens"], "202")

    def test_save_results_to_csv_defaults_to_cwd(self) -> None:
        results = [
            core.ExtractionResult(
                file_name="doc_a.pdf",
                pdf_path=Path("doc_a.pdf"),
                input_tokens=11,
                output=ExtractedDocument(summary="summary a", category="Other"),
                output_tokens=101,
                model="gpt-5-nano",
                status="COMPLETE",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                out_path = core.save_results_to_csv(results)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(out_path.resolve(), (Path(tmp) / "extractai_results.csv").resolve())
            self.assertTrue(out_path.exists())

    def test_save_results_to_csv_with_empty_results_writes_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = core.save_results_to_csv([], output_dir=tmp)
            with out_path.open("r", encoding="utf-8") as handle:
                content = handle.read().strip()
            self.assertEqual(content, "id,model,status,input_tokens,output_tokens")


class BatchWorkflowTests(unittest.TestCase):
    def test_submit_directory_batch_creates_manifest(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", model="gpt-5-nano", max_input_tokens=500)
        fake_client = _FakeBatchClient()

        with tempfile.TemporaryDirectory() as tmp:
            with patch("extractai.core.list_pdf_files", return_value=[Path("a.pdf")]), patch(
                "extractai.core.extract_text_from_pdf", return_value="short text"
            ), patch("extractai.core.make_token_counter", return_value=len):
                submission = core.submit_directory_batch(
                    config=cfg,
                    schema=ExtractedDocument,
                    prompt="prompt",
                    output_dir=tmp,
                    client=fake_client,  # type: ignore[arg-type]
                )

            self.assertEqual(submission.batch_id, "batch_123")
            self.assertEqual(submission.status, "SUBMITTED")
            self.assertTrue(submission.manifest_path.exists())

            manifest = json.loads(submission.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["batch_id"], "batch_123")
            self.assertEqual(manifest["schema_fields"], ["summary", "category"])
            self.assertEqual(len(manifest["documents"]), 1)
            self.assertEqual(manifest["documents"][0]["status"], "PENDING")

            with submission.input_jsonl_path.open("r", encoding="utf-8") as handle:
                task = json.loads(handle.readline())
            self.assertNotIn("temperature", task["body"])
            self.assertEqual(task["body"]["response_format"]["type"], "json_object")
            self.assertIn("json", task["body"]["messages"][0]["content"].lower())

    def test_collect_batch_results_builds_complete_and_failed_rows(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", model="gpt-5-nano", max_input_tokens=500)
        fake_client = _FakeBatchClient()

        output_line = {
            "custom_id": "doc-0",
            "response": {
                "status_code": 200,
                "body": {
                    "model": "gpt-5-nano",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({"summary": "ok", "category": "Other"})
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 12},
                },
            },
            "error": None,
        }
        error_line = {
            "custom_id": "doc-1",
            "response": None,
            "error": {"code": "network", "message": "temporary failure"},
        }
        fake_client._output_by_file_id["file_output_1"] = json.dumps(output_line) + "\n"
        fake_client._output_by_file_id["file_error_1"] = json.dumps(error_line) + "\n"

        with tempfile.TemporaryDirectory() as tmp:
            with patch("extractai.core.list_pdf_files", return_value=[Path("a.pdf"), Path("b.pdf")]), patch(
                "extractai.core.extract_text_from_pdf", side_effect=["text a", "text b"]
            ), patch("extractai.core.make_token_counter", return_value=len):
                submission = core.submit_directory_batch(
                    config=cfg,
                    schema=ExtractedDocument,
                    prompt="prompt",
                    output_dir=tmp,
                    client=fake_client,  # type: ignore[arg-type]
                )

            results, csv_path = core.collect_batch_results(
                submission_id=submission.submission_id,
                schema=ExtractedDocument,
                output_dir=tmp,
                client=fake_client,  # type: ignore[arg-type]
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].status, "COMPLETE")
            self.assertEqual(results[0].output.summary, "ok")
            self.assertEqual(results[0].output_tokens, 12)
            self.assertEqual(results[1].status, "FAILED")
            self.assertEqual(results[1].output.summary, "")
            self.assertTrue(csv_path.exists())

            batch_dir = Path(tmp) / ".extractai" / "batches" / submission.submission_id
            self.assertTrue((batch_dir / "batch_status.json").exists())
            self.assertTrue((batch_dir / "output.jsonl").exists())
            self.assertTrue((batch_dir / "error.jsonl").exists())

    def test_get_batch_status_for_local_only_submission(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", model="gpt-5-nano", max_input_tokens=1)

        with tempfile.TemporaryDirectory() as tmp:
            with patch("extractai.core.list_pdf_files", return_value=[Path("a.pdf")]), patch(
                "extractai.core.extract_text_from_pdf", return_value="long text"
            ), patch("extractai.core.make_token_counter", return_value=len):
                submission = core.submit_directory_batch(
                    config=cfg,
                    schema=ExtractedDocument,
                    prompt="prompt",
                    output_dir=tmp,
                    client=_FakeBatchClient(),  # type: ignore[arg-type]
                )

            status = core.get_batch_status(
                submission_id=submission.submission_id,
                output_dir=tmp,
            )
            self.assertEqual(status["status"], "completed")
            self.assertTrue(status["local_only"])

    def test_get_batch_status_uses_api_key_argument(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", model="gpt-5-nano", max_input_tokens=500)
        submission_client = _FakeBatchClient()
        status_client = _FakeBatchClient()

        with tempfile.TemporaryDirectory() as tmp:
            with patch("extractai.core.list_pdf_files", return_value=[Path("a.pdf")]), patch(
                "extractai.core.extract_text_from_pdf", return_value="short text"
            ), patch("extractai.core.make_token_counter", return_value=len):
                submission = core.submit_directory_batch(
                    config=cfg,
                    schema=ExtractedDocument,
                    prompt="prompt",
                    output_dir=tmp,
                    client=submission_client,  # type: ignore[arg-type]
                )

            with patch("extractai.core.OpenAI", return_value=status_client) as mock_openai:
                status = core.get_batch_status(
                    submission_id=submission.submission_id,
                    output_dir=tmp,
                    api_key="direct-api-key",
                )

            mock_openai.assert_called_once_with(api_key="direct-api-key")
            self.assertEqual(status["id"], "batch_123")

    def test_run_extraction_uses_sync_path_when_use_batch_false(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", use_batch=False)
        expected = [
            core.ExtractionResult(
                file_name="a.pdf",
                pdf_path=Path("a.pdf"),
                input_tokens=10,
                output=ExtractedDocument(summary="ok", category="Other"),
            )
        ]

        with patch("extractai.core.run_directory_extraction", return_value=expected) as mock_sync:
            result = core.run_extraction(config=cfg, schema=ExtractedDocument, prompt="prompt")

        self.assertEqual(result, expected)
        mock_sync.assert_called_once()

    def test_run_extraction_uses_batch_path_when_use_batch_true(self) -> None:
        cfg = core.ExtractAIConfig(pdf_dir="sample_data", use_batch=True)
        expected = core.BatchSubmission(
            submission_id="batch_123",
            batch_id="batch_123",
            input_file_id="file_1",
            batch_dir=Path("/tmp/batch_123"),
            manifest_path=Path("/tmp/batch_123/manifest.json"),
            input_jsonl_path=Path("/tmp/batch_123/batch_input.jsonl"),
            submitted_requests=1,
            skipped_requests=0,
            failed_requests=0,
            status="SUBMITTED",
        )

        with patch("extractai.core.submit_directory_batch", return_value=expected) as mock_batch:
            result = core.run_extraction(config=cfg, schema=ExtractedDocument, prompt="prompt")

        self.assertEqual(result, expected)
        mock_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
