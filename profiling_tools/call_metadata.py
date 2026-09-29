"""Diagnostic metadata comparison; never changes CPU alignment scores."""
import json

from .parser import normalize_input_dtypes, normalize_input_shapes


def shapes(value):
    if value is None or (isinstance(value, str) and value.strip().strip('"').lower() in {"", "n/a", "none", "null", "unknown"}):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = json.loads(normalize_input_shapes(value) or "[]")
    if not isinstance(value, (list, tuple)) or (value and all(not isinstance(v, (list, tuple, dict)) for v in value)):
        return json.loads(normalize_input_shapes(value) or "[]")
    return value


def dtypes(value):
    if value is None or str(value).strip().strip('"').lower() in {"", "n/a", "null", "unknown"}:
        return []
    value = str(value).replace("c10::", "").replace("at::", "")
    return [v or "none" for v in json.loads(normalize_input_dtypes(value) or "[]")]


def compare_metadata(npu, gpu, op_map=None):
    """Separate missing TensorList elements from actual shape conflicts.

    Scalar/zero-dimensional tensor [] is valid. A TensorList [] lacks element
    metadata and is not evidence that two non-empty tensor lists are equal.
    """
    differences, missing = [], []
    ns, gs = (shapes(op.raw_input_shapes or op.input_shapes) for op in (npu, gpu))
    nd, gd = (dtypes(op.raw_input_dtypes or op.input_dtypes) for op in (npu, gpu))
    if not ns or not gs:
        missing.append("input_shapes")
    elif len(ns) != len(gs):
        differences.append("input_shapes")
    else:
        for i, (a, b) in enumerate(zip(ns, gs)):
            tensor_list = (i < len(nd) and nd[i] == "tensorlist") or (i < len(gd) and gd[i] == "tensorlist")
            if tensor_list and (not a or not b):
                missing.append("input_shapes")
            elif a != b:
                differences.append("input_shapes")
    if not nd or not gd:
        missing.append("input_dtypes")
    elif nd != gd:
        differences.append("input_dtypes")
    if not npu.module_key or not gpu.module_key:
        missing.append("module")
    elif npu.module_key != gpu.module_key:
        differences.append("module")
    if npu.canonical_name != (op_map or {}).get(gpu.canonical_name, gpu.canonical_name):
        differences.append("name")
    return sorted(set(differences)), sorted(set(missing))
