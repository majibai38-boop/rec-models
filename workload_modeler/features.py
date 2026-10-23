"""Conservative logical-work extraction; never infer physical DRAM traffic."""
import json
import math


class ModelIssue(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def canonical_dtype(value):
    text = str(value).strip().lower()
    for prefix in ('torch.', 'c10::', 'at::', 'scalar_type::'):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return {'float':'fp32', 'float32':'fp32', 'float16':'fp16', 'half':'fp16',
            'bfloat16':'bf16', 'fp32':'fp32', 'fp16':'fp16', 'bf16':'bf16'}.get(text)


def normalize_dtypes_json(value):
    """Normalize known floating aliases; preserve parameter and unknown tokens."""
    if not value:
        return value
    try:
        values = json.loads(value)
    except (ValueError, TypeError):
        return value
    if not isinstance(values, list):
        return value
    names = {'fp16':'float16', 'bf16':'bfloat16', 'fp32':'float32'}
    return json.dumps([names.get(canonical_dtype(t), t) for t in values], separators=(',', ':'))


def extract_features(unit):
    op = unit['op']
    if op not in {'aten::mm', 'aten::bmm', 'aten::matmul', 'aten::add', 'aten::sub', 'aten::mul', 'aten::relu'}:
        raise ModelIssue('unsupported_operator', '该算子/融合路径暂无通用模型，请提供同签名双侧校准')
    try:
        shapes = json.loads(unit['input_shapes'])
        types = json.loads(unit['input_dtypes'])
    except (ValueError, TypeError):
        raise ModelIssue('metadata', 'Shape/dtype缺失或不是有效的规范化元数据')
    if not isinstance(shapes, list) or not isinstance(types, list) or not shapes or len(shapes) != len(types):
        raise ModelIssue('metadata', '输入Shape与dtype数量不一致或缺失')
    if any(not isinstance(s, list) or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in s) for s in shapes):
        raise ModelIssue('metadata', '动态/非正维度或无效Shape暂不支持')
    if any(str(t).lower() in {'scalar', 'scalarlist', 'tensorlist'} for t in types):
        raise ModelIssue('parameters', '包含Scalar/TensorList等参数，需要按算子签名解释，不能视为未知Tensor精度或直接删除')
    dtypes = [canonical_dtype(t) for t in types]
    if None in dtypes or len(set(dtypes)) != 1:
        raise ModelIssue('dtype', '混合精度、整数/布尔或未知dtype未被当前浮点模型覆盖')
    dtype = dtypes[0]
    size = 4 if dtype == 'fp32' else 2
    op = unit['op']
    if op in {'aten::mm', 'aten::bmm', 'aten::matmul'}:
        if unit.get('metadata_source') == 'npu_kernel':
            raise ModelIssue('implementation', 'NPU MatMul kernel缺少明确转置/布局语义；不能仅凭类型名当作aten::mm')
        if len(shapes) != 2 or len(shapes[0]) not in (2, 3) or len(shapes[1]) != len(shapes[0]):
            raise ModelIssue('unsupported_shape', '矩阵模型仅支持两个2D/同batch 3D输入；广播/转置需要专用模型')
        a, b = shapes
        if a[-1] != b[-2] or (len(a) == 3 and a[0] != b[0]):
            raise ModelIssue('unsupported_shape', '矩阵维度不兼容，不能推断转置或广播')
        batch = a[0] if len(a) == 3 else 1
        m, k, n = a[-2], a[-1], b[-1]
        work = 2 * batch * m * n * k
        logical_bytes = (math.prod(a) + math.prod(b) + batch*m*n) * size
        return dict(family='matrix', dtype=dtype, flops=work, logical_bytes=logical_bytes,
                    output_shape=a[:-2] + [m, n], output_basis='inferred_from_cpu_op')
    if op in {'aten::add', 'aten::sub', 'aten::mul', 'aten::relu'}:
        expected = 1 if op == 'aten::relu' else 2
        if len(shapes) != expected or any(s != shapes[0] for s in shapes) or not shapes[0]:
            raise ModelIssue('unsupported_shape', '逐元素模型仅支持完整同Shape输入，标量/广播/融合待校准')
        # NPU kernel outputs, if provided, must agree with the assumed non-fused path.
        if unit.get('output_shapes'):
            try:
                outputs = json.loads(unit['output_shapes'])
            except (ValueError, TypeError):
                raise ModelIssue('metadata', '输出Shape无效')
            if outputs != [shapes[0]]:
                raise ModelIssue('implementation', '已记录输出Shape与逐元素模型不一致，需要核验融合或布局')
        if unit.get('output_dtypes'):
            try:
                out_types = json.loads(unit['output_dtypes'])
            except (ValueError, TypeError):
                raise ModelIssue('metadata', '输出dtype无效')
            if not isinstance(out_types, list) or len(out_types) != 1 or canonical_dtype(out_types[0]) != dtype:
                raise ModelIssue('dtype', '输出dtype与输入不一致，不能忽略随路类型转换')
        n = math.prod(shapes[0])
        return dict(family='vector', dtype=dtype, flops=n, logical_bytes=(expected+1)*n*size,
                    output_shape=shapes[0], output_basis='inferred_and_checked_if_recorded')
    raise ModelIssue('unsupported_operator', '该算子/融合路径暂无通用模型，请提供同签名双侧校准')


SUGGESTIONS = {
    'parameters': '按算子签名区分Tensor、标量和TensorList，补充广播/alpha/beta等专用语义；无需仅为此重新采集dtype。',
    'metadata': '检查采集是否启用Shape/dtype记录，确认CSV列格式；不要用其他调用的元数据补齐。',
    'dtype': '核对输入/输出dtype及标量参数，检查混合精度与Cast路径；提供对应精度的实测校准。',
    'implementation': '核对转置、布局和融合实现，优先提供同路径微基准；不能按名称认定语义等价。',
    'unsupported_shape': '核对广播、标量、batch维度及布局；为该Shape补充专用模型或双侧校准。',
    'unsupported_operator': '优先为高耗时类型补充专用模型或同签名实测校准，不要按总峰值统一折算。',
    'missing_spec': '在规格库补齐对应精度算力/显存带宽及来源；升级平台引用后复用输入重跑。',
    'estimated_spec': '优先补充实测或可靠规格；仅在接受估算风险时启用estimated规格。',
    'dense_basis': '确认矩阵峰值是稠密还是稀疏口径，核验后创建新规格版本；不要直接勾选稠密。',
    'precision': '确认源/目标矩阵计算模式与精度策略，再决定是否允许跨TF32/HF32模式比较。',
    'activity': '通信和内存活动需要对应链路/传输条件的独立模型或校准。',
    'zero_time': '检查计时分辨率和采集记录；零时长不能构造性能比。',
}


def diagnostics(rows, total):
    """Every unmodelled unit contributes to exactly one reason bucket."""
    buckets = {}
    for r in rows:
        if r['predicted_us'] is not None:
            continue
        code = r['reason_code']
        b = buckets.setdefault(code, dict(code=code, source_us=0., unit_count=0, kernel_count=0,
            operators=set(), examples=[], suggestion=SUGGESTIONS.get(code, '核验调用证据后补充校准。')))
        b['source_us'] += r['source_us']
        b['unit_count'] += 1
        b['kernel_count'] += r['kernel_count']
        b['operators'].add(r['op'])
        if len(b['examples']) < 3:
            b['examples'].append(dict(signature=r['signature'], op=r['op'], reason=r['reason']))
    for b in buckets.values():
        b['operators'] = sorted(b['operators'])
        b['source_pct'] = 100*b['source_us']/total if total else None
    return sorted(buckets.values(), key=lambda b: b['source_us'], reverse=True)
