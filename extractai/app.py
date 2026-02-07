from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import streamlit as st
from pydantic import BaseModel, create_model

# Support running via `streamlit run extractai/extractai-app.py` from repo root
# without requiring `pip install -e .`.
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from extractai.core import (
    ExtractAIConfig,
    build_prompt,
    list_openai_models,
    run_directory_extraction,
    save_results_to_csv,
)


TYPE_MAP: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "datetime": datetime,
}

TYPE_OPTIONS = ["str", "int", "float", "bool", "datetime", "list"]
DEFAULT_SCHEMA_ROWS = [
    {"name": "summary", "type": "str", "choices": ""},
    {
        "name": "category",
        "type": "str",
        "choices": "Financial, Research, Government, Other",
    },
]

DEFAULT_PROMPT = (
    "Extract two fields from the document text: summary and category. "
    "Category must be exactly one of: Financial, Research, Government, Other. "
    "If uncertain, choose the best matching option."
)
DEFAULT_MODEL = "gpt-5-nano"
SCHEMA_ROWS_STATE_KEY = "schema_rows"
SCHEMA_TABLE_LAYOUT = [2.6, 1.3, 3.1, 1.0]
LAST_ROWS_STATE_KEY = "last_results_rows"
LAST_CSV_STATE_KEY = "last_results_csv_path"
DIRECTORY_PICKER_LAYOUT = [8.4, 1.6]
IO_SECTION_LAYOUT = [2.2, 1]


def _runtime_dependency_error() -> str | None:
    try:
        import pdfplumber  # noqa: F401
    except Exception:
        return (
            "Missing dependency: `pdfplumber`.\n\n"
            "Install runtime dependencies in the same Python environment that runs Streamlit:\n"
            f"- Python: `{sys.executable}`\n"
            "- Command: `python -m pip install -e '.[demo]'`"
        )
    return None


def _inject_background_styles() -> None:
    st.markdown(
        """
        <style>
        html, body {
            min-height: 100%;
            background: #f7f9fc;
        }

        .stApp {
            background: transparent;
        }

        [data-testid="stAppViewContainer"] {
            position: relative;
            background: transparent;
        }

        [data-testid="stAppViewContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background:
                radial-gradient(circle at 10% 16%, rgba(173, 216, 255, 0.46) 0%, rgba(173, 216, 255, 0) 42%),
                radial-gradient(circle at 88% 14%, rgba(203, 176, 255, 0.40) 0%, rgba(203, 176, 255, 0) 41%),
                radial-gradient(circle at 86% 84%, rgba(255, 192, 203, 0.36) 0%, rgba(255, 192, 203, 0) 43%),
                radial-gradient(circle at 16% 86%, rgba(255, 244, 179, 0.42) 0%, rgba(255, 244, 179, 0) 42%),
                linear-gradient(135deg, #f8fbff 0%, #fff8fd 54%, #fffef7 100%);
            background-repeat: no-repeat;
            background-size: cover;
        }

        [data-testid="stAppViewContainer"] > .main {
            position: relative;
            z-index: 1;
            background: transparent;
        }

        [data-testid="stAppViewContainer"] .block-container {
            position: relative;
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(211, 222, 236, 0.9);
            border-radius: 18px;
            box-shadow: 0 18px 44px rgba(42, 65, 95, 0.12);
            padding: 1.25rem 1.5rem 1.8rem 1.5rem;
            max-width: 1360px;
            width: calc(100% - 3.0rem);
            margin-left: auto;
            margin-right: auto;
            margin-top: 1.0rem;
            margin-bottom: 1.0rem;
            backdrop-filter: saturate(1.05);
        }

        [data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Light-theme fallback for cloud deployments */
        [data-testid="stAppViewContainer"] .block-container h1,
        [data-testid="stAppViewContainer"] .block-container h2,
        [data-testid="stAppViewContainer"] .block-container h3,
        [data-testid="stAppViewContainer"] .block-container h4,
        [data-testid="stAppViewContainer"] .block-container p,
        [data-testid="stAppViewContainer"] .block-container label,
        [data-testid="stAppViewContainer"] .block-container div[data-testid="stMarkdownContainer"],
        [data-testid="stAppViewContainer"] .block-container div[data-testid="stCaptionContainer"] {
            color: #1f2937 !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1px solid #c8d3e3 !important;
            border-radius: 0.55rem !important;
            box-shadow: none !important;
        }

        div[data-baseweb="input"] > div:hover,
        div[data-baseweb="textarea"] > div:hover,
        div[data-baseweb="select"] > div:hover {
            border-color: #9bb0cf !important;
        }

        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="textarea"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {
            border-color: #1d4ed8 !important;
            box-shadow: 0 0 0 1px #1d4ed8 !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="input"] textarea,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div[role="combobox"] {
            color: #111827 !important;
            background: transparent !important;
            font-size: 0.84rem !important;
        }

        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="input"] textarea::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder,
        div[data-baseweb="select"] input::placeholder {
            font-size: 0.84rem !important;
        }

        button[id*="input_dir_browse_button"],
        button[id*="output_dir_browse_button"] {
            width: 100% !important;
            min-height: 2.05rem !important;
            padding: 0.10rem 0.38rem !important;
            line-height: 1 !important;
            margin-top: 0.06rem !important;
            margin-left: 0 !important;
            margin-bottom: 0 !important;
            border-radius: 0.55rem !important;
            white-space: nowrap !important;
            z-index: 3 !important;
        }

        button[id*="input_dir_browse_button"] p,
        button[id*="output_dir_browse_button"] p {
            font-size: 0.67rem !important;
            line-height: 1 !important;
            margin: 0 !important;
            white-space: nowrap !important;
            word-break: normal !important;
        }

        /* Keep Execute button text readable */
        [data-testid="stButton"] button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            color: #ffffff !important;
        }

        [data-testid="stButton"] button[kind="primary"] p,
        [data-testid="stFormSubmitButton"] button[kind="primary"] p {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _open_file_with_default_app(path: Path) -> None:
    target = path.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Output file not found: {target}")

    system = platform.system().lower()

    if system == "windows":
        os.startfile(str(target))  # type: ignore[attr-defined]
        return

    command = ["open", str(target)] if system == "darwin" else ["xdg-open", str(target)]
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _pick_directory_native(*, initial_dir: Path, title: str) -> str | None:
    system = platform.system().lower()
    safe_title = title.replace('"', '\\"')
    safe_initial_dir = str(initial_dir).replace('"', '\\"')

    if system == "darwin":
        script = f'POSIX path of (choose folder with prompt "{safe_title}")'
        try:
            completed = subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )
            selected = completed.stdout.strip()
            return selected or None
        except subprocess.CalledProcessError as exc:
            # User-cancelled picker returns non-zero.
            error_text = (exc.stderr or "") + (exc.stdout or "")
            if "User canceled" in error_text:
                return None
            raise RuntimeError(f"Native folder picker failed: {error_text.strip()}") from exc

    if system == "windows":
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            f'$dialog.Description = "{safe_title}"; '
            f'$dialog.SelectedPath = "{safe_initial_dir}"; '
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
            "Write-Output $dialog.SelectedPath }"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            check=False,
            capture_output=True,
            text=True,
        )
        selected = completed.stdout.strip()
        return selected or None

    # Linux / other: use zenity when available.
    zenity_path = shutil.which("zenity")
    if not zenity_path:
        return None

    completed = subprocess.run(
        [zenity_path, "--file-selection", "--directory", "--filename", f"{initial_dir}/"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        selected = completed.stdout.strip()
        return selected or None

    return None


def _resolve_initial_picker_dir(*, raw_value: str, default_dir: Path) -> Path:
    candidate = Path(raw_value.strip() or str(default_dir))
    return candidate if candidate.exists() else default_dir


def _native_picker_available() -> bool:
    system = platform.system().lower()
    if system in {"darwin", "windows"}:
        return True
    return shutil.which("zenity") is not None


def _directory_picker(*, label: str, key_prefix: str, default_dir: Path) -> str:
    path_key = f"{key_prefix}_path"

    if path_key not in st.session_state:
        st.session_state[path_key] = str(default_dir)

    st.markdown(label)
    field_col, button_col = st.columns(DIRECTORY_PICKER_LAYOUT, gap="small")
    can_use_native_picker = _native_picker_available()
    with field_col:
        typed_value = st.text_input(label, value=str(st.session_state[path_key]), label_visibility="collapsed")
        if typed_value != st.session_state[path_key]:
            st.session_state[path_key] = typed_value
        if not can_use_native_picker:
            st.caption("Browse is unavailable in this environment. Enter a directory path manually.")
    with button_col:
        if st.button(
            "Browse",
            key=f"{key_prefix}_browse_button",
            use_container_width=True,
            disabled=not can_use_native_picker,
            help=None if can_use_native_picker else "Unavailable on this hosted environment.",
        ):
            initial_dir = _resolve_initial_picker_dir(
                raw_value=str(st.session_state[path_key]),
                default_dir=default_dir,
            )
            try:
                picked = _pick_directory_native(initial_dir=initial_dir, title=label)
                if picked:
                    st.session_state[path_key] = picked
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not open native folder picker: {exc}")

    return str(st.session_state[path_key])


def _discover_dirs(base_dir: Path, *, max_depth: int = 2) -> list[str]:
    candidates: set[str] = set()

    for root, dirs, _files in _walk_depth(base_dir, max_depth=max_depth):
        root_path = Path(root)
        if any(root_path.glob("*.pdf")):
            candidates.add(str(root_path))
        for name in dirs:
            full = root_path / name
            if any(full.glob("*.pdf")):
                candidates.add(str(full))

    return sorted(candidates)


def _walk_depth(base_dir: Path, *, max_depth: int):
    base_depth = len(base_dir.resolve().parts)
    for root, dirs, files in os.walk(base_dir):
        depth = len(Path(root).resolve().parts) - base_depth
        if depth >= max_depth:
            dirs[:] = []
        yield root, dirs, files


def _new_schema_row(
    *,
    name: str = "",
    field_type: str = "str",
    choices: str = "",
) -> dict[str, Any]:
    return {
        "row_id": uuid4().hex[:8],
        "name": name,
        "type": field_type,
        "choices": choices,
    }


def _default_schema_rows() -> list[dict[str, Any]]:
    return [
        _new_schema_row(
            name=row["name"],
            field_type=row["type"],
            choices=row["choices"],
        )
        for row in DEFAULT_SCHEMA_ROWS
    ]


def _ensure_schema_rows() -> list[dict[str, Any]]:
    rows = st.session_state.get(SCHEMA_ROWS_STATE_KEY)
    if isinstance(rows, list) and rows:
        return rows

    initialized_rows = _default_schema_rows()
    st.session_state[SCHEMA_ROWS_STATE_KEY] = initialized_rows
    return initialized_rows


def _split_choices(raw: str) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


def _schema_rows_to_display(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    display_rows: list[dict[str, Any]] = []
    for row in rows:
        choices = _split_choices(str(row.get("choices", "")))
        entry = {
            "name": str(row.get("name", "")).strip(),
            "type": str(row.get("type", "")).strip(),
        }
        if choices:
            entry["choices"] = choices
        display_rows.append(entry)
    return display_rows


def _build_schema_from_rows(rows: list[dict[str, Any]]) -> type[BaseModel]:
    if not rows:
        raise ValueError("Schema must include at least one column")

    field_defs: dict[str, tuple[Any, Any]] = {}
    for idx, row in enumerate(rows):
        name = str(row.get("name", "")).strip()
        type_name = str(row.get("type", "")).strip().lower()
        choices = _split_choices(str(row.get("choices", "")))

        if not name:
            raise ValueError(f"Schema row {idx + 1} is missing a column name")
        if name in field_defs:
            raise ValueError(f"Duplicate column name in schema: {name}")
        if type_name not in TYPE_OPTIONS:
            raise ValueError(f"Unsupported type '{type_name}' for column '{name}'")

        if type_name == "str":
            annotation = Literal[tuple(choices)] if choices else str
        elif type_name == "list":
            annotation = list[Literal[tuple(choices)]] if choices else list[str]
        else:
            annotation = TYPE_MAP[type_name]

        field_defs[name] = (annotation, ...)

    return create_model("FrontendExtractedDocument", **field_defs)


def _render_schema_builder() -> list[dict[str, Any]]:
    rows = _ensure_schema_rows()

    header_cols = st.columns(SCHEMA_TABLE_LAYOUT)
    header_cols[0].markdown("**Column name**")
    header_cols[1].markdown("**Type**")
    header_cols[2].markdown("**Allowed values (optional)**")
    header_cols[3].markdown("**Delete**")

    updated_rows: list[dict[str, Any]] = []
    delete_row_id: str | None = None

    for row in rows:
        row_id = str(row.get("row_id") or uuid4().hex[:8])
        default_type = str(row.get("type", "str"))
        default_index = TYPE_OPTIONS.index(default_type) if default_type in TYPE_OPTIONS else 0

        cols = st.columns(SCHEMA_TABLE_LAYOUT)
        name = cols[0].text_input(
            "Column name",
            value=str(row.get("name", "")),
            key=f"schema_name_{row_id}",
            label_visibility="collapsed",
            placeholder="summary",
        )
        field_type = cols[1].selectbox(
            "Type",
            options=TYPE_OPTIONS,
            index=default_index,
            key=f"schema_type_{row_id}",
            label_visibility="collapsed",
        )
        choices = cols[2].text_input(
            "Allowed values",
            value=str(row.get("choices", "")),
            key=f"schema_choices_{row_id}",
            label_visibility="collapsed",
            placeholder="for list: valueA,valueB,valueC",
        )
        if cols[3].button("X", key=f"schema_delete_{row_id}", use_container_width=True):
            delete_row_id = row_id

        updated_rows.append(
            {
                "row_id": row_id,
                "name": name,
                "type": field_type,
                "choices": choices,
            }
        )

    st.session_state[SCHEMA_ROWS_STATE_KEY] = updated_rows

    if delete_row_id is not None:
        remaining = [row for row in updated_rows if row["row_id"] != delete_row_id]
        if not remaining:
            remaining = [_new_schema_row(name="summary", field_type="str", choices="")]
        st.session_state[SCHEMA_ROWS_STATE_KEY] = remaining
        st.rerun()

    add_col, _spacer = st.columns([1.2, 4])
    with add_col:
        if st.button("+ Add column", use_container_width=True):
            st.session_state[SCHEMA_ROWS_STATE_KEY] = [*updated_rows, _new_schema_row()]
            st.rerun()

    return st.session_state[SCHEMA_ROWS_STATE_KEY]


def _results_to_rows(results: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": result.file_name,
            **result.output.model_dump(),
            "model": result.model,
            "status": result.status,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "error": result.error,
        }
        for result in results
    ]


@st.cache_data(show_spinner=False)
def _load_models_cached(api_key: str | None) -> list[str]:
    key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
    return list_openai_models(api_key=key)


def _load_models_safe(api_key: str) -> tuple[list[str], str | None]:
    try:
        return _load_models_cached(api_key), None
    except Exception as exc:
        return [], str(exc)


def _render_io_section(*, cwd: Path) -> tuple[str, str, int]:
    discovered_pdf_dirs = _discover_dirs(cwd)
    default_input = (
        Path(discovered_pdf_dirs[0])
        if discovered_pdf_dirs
        else (cwd / "sample_data")
    )
    default_output = cwd / "outputs"

    st.subheader("1. Input / Output")
    st.caption("Use Browse to open your OS folder picker (Finder on macOS, Explorer on Windows).")

    left_col, right_col = st.columns(IO_SECTION_LAYOUT, gap="large")
    with left_col:
        input_dir = _directory_picker(
            label="Input PDF directory",
            key_prefix="input_dir",
            default_dir=default_input,
        )
        output_dir = _directory_picker(
            label="Output directory for CSV",
            key_prefix="output_dir",
            default_dir=default_output,
        )

    with right_col:
        st.markdown("**Max input tokens**")
        max_input_tokens = st.number_input(
            "Max input tokens",
            min_value=1,
            value=50_000,
            step=1_000,
            help="Documents that exceed this token limit will be marked SKIPPED.",
            label_visibility="collapsed",
        )

    return input_dir, output_dir, int(max_input_tokens)


def _render_model_section() -> tuple[str, str]:
    st.subheader("2. Model")
    model_col, key_col = st.columns([2, 5])

    with key_col:
        api_key = st.text_input(
            "OpenAI API key (optional if already set in env)",
            type="password",
            placeholder="sk-...",
        )

    models, model_error = _load_models_safe(api_key)
    with model_col:
        if models:
            default_index = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
            model = st.selectbox("Model (searchable)", options=models, index=default_index)
        else:
            if model_error:
                st.warning(f"Could not load model list automatically: {model_error}")
            model = st.text_input("Model", value=DEFAULT_MODEL)

    return model, api_key


def _render_prompt_schema_section() -> tuple[str, list[dict[str, Any]]]:
    st.subheader("3. Prompt and Schema")
    prompt_col, schema_col = st.columns(2)

    with prompt_col:
        prompt_text = st.text_area("Prompt", value=DEFAULT_PROMPT, height=220)

    with schema_col:
        st.caption(
            "Define output columns. For `str` or `list`, optional allowed values "
            "can be provided as comma-separated text."
        )
        schema_rows = _render_schema_builder()
        with st.expander("Generated schema preview", expanded=False):
            st.code(json.dumps(_schema_rows_to_display(schema_rows), indent=2), language="json")

    return prompt_text, schema_rows


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("status")) for row in rows))


def _render_results(rows: list[dict[str, Any]], csv_path: Path) -> None:
    st.success(f"Completed. CSV saved to: {csv_path}")
    open_col, _spacer = st.columns([1.5, 4])
    with open_col:
        if st.button("Open Output CSV", key="open_output_csv_btn", use_container_width=True):
            try:
                _open_file_with_default_app(csv_path)
                st.info(f"Opened: {csv_path}")
            except Exception as exc:
                st.error(f"Could not open CSV automatically: {exc}")

    st.write("Status counts:", _status_counts(rows))
    st.dataframe(rows, use_container_width=True)

    failed_rows = [row for row in rows if row.get("status") == "FAILED"]
    if not failed_rows:
        return

    st.warning("One or more files failed. See errors below.")
    st.dataframe(
        [
            {
                "id": row.get("id"),
                "status": row.get("status"),
                "error": row.get("error"),
            }
            for row in failed_rows
        ],
        use_container_width=True,
    )


def _execute_extraction(
    *,
    input_dir: str,
    output_dir: str,
    max_input_tokens: int,
    model: str,
    api_key: str,
    prompt_text: str,
    schema_rows: list[dict[str, Any]],
) -> None:
    dependency_error = _runtime_dependency_error()
    if dependency_error:
        st.error(dependency_error)
        return

    prompt = build_prompt(prompt_text)
    schema = _build_schema_from_rows(schema_rows)
    config = ExtractAIConfig(
        pdf_dir=input_dir,
        model=model,
        max_input_tokens=max_input_tokens,
        api_key=api_key.strip() or None,
        use_batch=False,
    )
    progress_placeholder = st.empty()

    def on_progress(processed: int, total: int, _path: Path) -> None:
        progress_placeholder.caption(f"Processing {processed}/{total} documents...")

    with st.spinner("Running extraction..."):
        results = run_directory_extraction(
            config=config,
            schema=schema,
            prompt=prompt,
            progress_callback=on_progress,
        )
        csv_path = save_results_to_csv(results, output_dir=output_dir)

    progress_placeholder.empty()
    st.session_state[LAST_ROWS_STATE_KEY] = _results_to_rows(results)
    st.session_state[LAST_CSV_STATE_KEY] = str(csv_path)


def main() -> None:
    st.set_page_config(page_title="ExtractAI App", layout="wide")
    _inject_background_styles()
    st.title("ExtractAI App")
    st.caption("A web app that uses the OpenAI API to extract structured information from PDF files.")

    input_dir, output_dir, max_input_tokens = _render_io_section(cwd=Path.cwd())
    model, api_key = _render_model_section()
    prompt_text, schema_rows = _render_prompt_schema_section()

    st.subheader("4. Execute")
    if st.button("Execute", type="primary"):
        try:
            _execute_extraction(
                input_dir=input_dir,
                output_dir=output_dir,
                max_input_tokens=max_input_tokens,
                model=model,
                api_key=api_key,
                prompt_text=prompt_text,
                schema_rows=schema_rows,
            )
        except Exception as exc:
            st.error(f"Execution failed: {exc}")

    cached_rows = st.session_state.get(LAST_ROWS_STATE_KEY)
    cached_csv_path = st.session_state.get(LAST_CSV_STATE_KEY)
    if isinstance(cached_rows, list) and isinstance(cached_csv_path, str) and cached_csv_path:
        _render_results(cached_rows, Path(cached_csv_path))


if __name__ == "__main__":
    main()
