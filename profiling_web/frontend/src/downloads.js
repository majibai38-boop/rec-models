export const exportFiles = [
  ['npu_kernel_details.csv', 'NPU kernel明细（所选范围）'],
  ['npu_op_statistic.csv', 'NPU算子统计（所选范围）'],
  ['gpu_kerenel_details.csv', 'GPU kernel明细（所选范围）'],
  ['gpu_op_statistic.csv', 'GPU kernel统计（所选范围）'],
  ['npu_trace.json', 'NPU Trace JSON'],
  ['gpu_trace.json', 'GPU Trace JSON'],
]
export const visibleDownloads = reports => exportFiles.filter(([name]) => (reports || []).includes(name))
export const inputDownloads = inputs => [
  ['gpu_trace', 'GPU Trace', 'gpu.json'],
  ['npu_trace', 'NPU Trace', 'npu.json'],
  ['npu_kernel_details', 'NPU kernel明细', 'kernel_details.csv'],
].filter(([role]) => inputs?.[role]).map(([role,label,fallback]) => ({role,label,name:inputs[role].name || fallback}))
