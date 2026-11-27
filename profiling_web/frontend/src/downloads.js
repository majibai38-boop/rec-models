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
  ['baseline_details', '基线 NPU kernel 明细', 'baseline.csv'],
  ['candidate_details', '对比 NPU kernel 明细', 'candidate.csv'],
].filter(([role]) => inputs?.[role]).map(([role,label,fallback]) => ({role,label,name:inputs[role].name || fallback}))

export function reportDownloads(job) {
  const pair = job.mode === 'npu_pair', reports = job.summary?.reports || []
  const excel = pair ? 'npu_pair.xlsx' : 'npu_gpu_op_statistic.xlsx'
  return {excel: reports.includes(excel) ? excel : null, inputs:inputDownloads(job.inputs),
    more:pair ? [['npu_pair.json','完整对比结果（JSON）']].filter(([f])=>reports.includes(f)) :
      [...visibleDownloads(reports), ...[['call_tables.xlsx','严格调用评审 Excel'],['op_type_comparison.xlsx','类型＋调用方聚合 Excel'],
        ['op_type_comparison.md','类型＋调用方聚合报告']].filter(([f])=>reports.includes(f))]}
}
