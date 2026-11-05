"""Shape-based parameter candidates, never an exact model/state_dict count."""
import json
import math

from .features import canonical_dtype


# Argument positions follow the CPU aten signatures, not device kernel layouts.
ROLES = {
    'aten::linear': ('linear', [(1, 'weight', 'dense'), (2, 'bias', 'dense')]),
    'aten::conv1d': ('conv', [(1, 'weight', 'dense'), (2, 'bias', 'dense')]),
    'aten::conv2d': ('conv', [(1, 'weight', 'dense'), (2, 'bias', 'dense')]),
    'aten::conv3d': ('conv', [(1, 'weight', 'dense'), (2, 'bias', 'dense')]),
    'aten::convolution': ('conv', [(1, 'weight', 'dense'), (2, 'bias', 'dense')]),
    'aten::_convolution': ('conv', [(1, 'weight', 'dense'), (2, 'bias', 'dense')]),
    'aten::embedding': ('embedding', [(0, 'weight', 'embedding')]),
    'aten::embedding_bag': ('embedding', [(0, 'weight', 'embedding')]),
    'aten::_embedding_bag': ('embedding', [(0, 'weight', 'embedding')]),
    'aten::layer_norm': ('layer_norm', [(2, 'weight', 'dense'), (3, 'bias', 'dense')]),
    'aten::native_layer_norm': ('layer_norm', [(2, 'weight', 'dense'), (3, 'bias', 'dense')]),
    'aten::batch_norm': ('batch_norm', [(1, 'weight', 'dense'), (2, 'bias', 'dense')]),
    'aten::native_batch_norm': ('batch_norm', [(1, 'weight', 'dense'), (2, 'bias', 'dense')]),
    'aten::group_norm': ('group_norm', [(2, 'weight', 'dense'), (3, 'bias', 'dense')]),
}
MATRIX = {'aten::mm': 1, 'aten::matmul': 1, 'aten::addmm': 2, 'aten::bmm': 1}


def candidates(ops, owner_indices):
    """Use only CPU owners of selected device kernels and their ancestors."""
    selected = set()
    for index in owner_indices:
        while index is not None and index not in selected:
            selected.add(index)
            index = ops[index].parent_index
    groups = {}
    missing = 0
    for index in sorted(selected):
        op = ops[index]
        name = op.canonical_name
        if name not in ROLES and name not in MATRIX:
            continue
        ancestor = op.parent_index
        skip = False
        while ancestor is not None:
            parent = ops[ancestor]
            if parent.canonical_name in ROLES or 'backward' in parent.event.name.lower():
                skip = True
                break
            ancestor = parent.parent_index
        if skip:
            continue
        try:
            shapes = json.loads(op.input_shapes)
            dtypes = json.loads(op.input_dtypes or '[]')
            if not isinstance(shapes, list) or not isinstance(dtypes, list):
                raise ValueError()
        except (ValueError, TypeError):
            missing += 1
            continue
        family, roles = ROLES.get(name, ('matrix', [(MATRIX.get(name), 'rhs', 'ambiguous')]))
        valid = False
        for slot, role, kind in roles:
            shape = shapes[slot] if slot < len(shapes) else None
            if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 for n in shape):
                continue
            if role in {'weight', 'rhs'} and family in {'linear', 'conv', 'embedding', 'matrix'} and len(shape) < 2:
                continue
            dtype = str(dtypes[slot]) if slot < len(dtypes) else 'unknown'
            dtype = canonical_dtype(dtype) or dtype
            module = op.module_key or op.module_path or ''
            key = (module, family, role, tuple(shape), dtype, kind)
            row = groups.setdefault(key, dict(kind=kind, family=family, role=role, shape=shape,
                dtype=dtype, module=module, numel=math.prod(shape), calls=0, op=name,
                example_event_id=op.event.index,
                dedup_basis='module_shape' if module else 'shape_only'))
            row['calls'] += 1
            valid = True
        if not valid:
            missing += 1
    rows = sorted(groups.values(), key=lambda row: (-row['numel'], row['op'], row['role']))
    def total(kind):
        items = [r['numel'] for r in rows if r['kind'] == kind]
        return sum(items) if items else None
    dense, embedding = total('dense'), total('embedding')
    return dict(version=1, status='candidates' if rows else 'unavailable', exact_parameters=None,
        candidate_parameters=(dense or 0)+(embedding or 0) if dense is not None or embedding is not None else None,
        dense_candidates=dense, embedding_candidates=embedding, ambiguous_matrix_elements=total('ambiguous'),
        missing_shape_calls=missing, groups=rows, scope='selected_device_kernel_cpu_owners_and_ancestors',
        note='候选参数量：按模块（若有）、算子族、权重角色、Shape及dtype去重。相同Shape的不同层可能被合并，'
             '共享权重或不同精度副本也可能重复；不是精确参数量或上下界。Embedding按完整表Shape计数，'
             '不是本步激活行数，也不代表稀疏梯度配置。矩阵右输入可能是激活，单列且不计入候选合计。'
             '未执行分支、融合权重、部分归一化及外置Embedding可能缺失；精确总量需模型/state_dict。')
