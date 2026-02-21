from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its content as a dictionary.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed YAML content as a dictionary. If the file is empty,
        an empty dictionary is returned.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    TypeError
        If the YAML root object is not a mapping (dict).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping (dict). Got: {type(data)}")
    return data


def dump_yaml(obj: Any) -> str:
    """
    Serialize a Python object into a YAML-formatted string.

    Parameters
    ----------
    obj : Any
        Python object to serialize.

    Returns
    -------
    str
        YAML string representation of the object.
    """
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)