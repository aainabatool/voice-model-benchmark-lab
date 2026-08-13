"""Dataset manifest schema.

A manifest is a small JSON file describing a dataset: its identity/version
(DatasetRef) plus the list of test cases it contains. See
tests/fixtures/tiny_dataset.json for an example, and spec section 9 for the
test case schema this is built from.
"""
from __future__ import annotations

from pydantic import BaseModel

from voice_benchmark.core.models import DatasetRef, TestCase


class DatasetManifest(BaseModel):
    dataset: DatasetRef
    test_cases: list[TestCase]
