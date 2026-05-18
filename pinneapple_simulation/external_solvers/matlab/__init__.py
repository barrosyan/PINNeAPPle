"""MATLAB integration for PINNeAPPle.

Two execution modes:
  MATLABEngine  — direct in-process calls via the MATLAB Engine for Python API.
                  Fastest; requires a licensed MATLAB installation.
  runner        — subprocess-based execution via ``matlab -batch``.
                  Works in environments without the Engine API; uses .mat files
                  for data exchange.

Common utilities:
  mat_io        — load/save .mat files (scipy-based)
  mat_to_upd    — convert MATLAB solver output to a UPD PhysicalSample
"""
from .engine import MATLABEngine
from .runner import run_matlab_script, run_matlab_function
from .mat_io import load_mat, save_mat
from .field_reader import mat_to_upd

__all__ = [
    "MATLABEngine",
    "run_matlab_script",
    "run_matlab_function",
    "load_mat",
    "save_mat",
    "mat_to_upd",
]
