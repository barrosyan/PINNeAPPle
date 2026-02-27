"""Benchmarks for PINNeAPPle.

These scripts are designed to be runnable both:

- as modules: `python -m benchmarks.data_io_bench ...`
- as scripts: `python benchmarks/data_io_bench.py ...`

They include small dependency checks and provide actionable error messages when
optional runtime deps are missing.
"""

from ._latency import latency_summary, LatencyStats

__all__ = ["latency_summary", "LatencyStats"]