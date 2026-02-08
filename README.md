# ExtractAI

Extract structured data from local PDF files with OpenAI models.

## Install (GitHub)

Backend only:

```bash
python -m pip install "extractai @ git+https://github.com/ldicarlo1/extractai.git@main"
```

With frontend:

```bash
python -m pip install "extractai[demo] @ git+https://github.com/ldicarlo1/extractai.git@main"
```

## Run Frontend App

```bash
extractai-app
```

![ExtractAI App](images/ExtractAI%20App.png)

## Configure

Set an API key in your shell:

```bash
export OPENAI_API_KEY="sk-..."
```

Or pass `api_key="sk-..."` in `ExtractAIConfig`.

## Quickstart

```python
from typing import Literal
from pydantic import BaseModel
from extractai import ExtractAIConfig, build_prompt, run_directory_extraction, save_results_to_csv

class ExtractedDocument(BaseModel):
    summary: str
    category: Literal["Financial", "Research", "Government", "Other"]

config = ExtractAIConfig(pdf_dir="sample_data", model="gpt-5-nano", max_input_tokens=50_000)
prompt = build_prompt("Extract summary and category. Category must be Financial, Research, Government, or Other.")

results = run_directory_extraction(config=config, schema=ExtractedDocument, prompt=prompt)
csv_path = save_results_to_csv(results, output_dir="outputs")
print(csv_path)
```

## Notebooks

- Main flow: [`extractai_basic_demo.ipynb`](extractai_basic_demo.ipynb)
- Batch flow: [`extractai_batch_demo.ipynb`](extractai_batch_demo.ipynb)
