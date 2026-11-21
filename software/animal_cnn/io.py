"""Serialization shared by training and the integer exporter."""

import json
from pathlib import Path
import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_mem(path, values, bits):
    mask = (1 << bits) - 1
    Path(path).write_text("".join(f"{int(v) & mask:0{bits//4}x}\n"
                                  for v in np.asarray(values).reshape(-1)), encoding="ascii")
