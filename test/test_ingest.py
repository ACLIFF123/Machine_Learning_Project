import os
from pathlib import Path
import pandas as pd
from src import ingest


def test_load_data_reads_jsonl(tmp_path: Path):
    """load_data should read a JSONL file into a DataFrame."""

    # Create a temporary JSONL file
    jsonl_content = "\n".join([
        '{"label": 1, "title": "Hello", "description": "World"}',
        '{"label": 2, "title": "Foo", "description": "Bar"}'
    ])

    jsonl_file = tmp_path / "sample.jsonl"
    jsonl_file.write_text(jsonl_content, encoding="utf-8")

    # Call the function
    df = ingest.load_data(jsonl_file)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    assert set(["label", "title", "description"]).issubset(df.columns)


