<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
const props = defineProps({ record: Object, evidence: Object })
const emit = defineEmits(['close'])
const closeButton = ref(null)
let previousFocus
function key(event) {
  if (event.key === 'Escape') emit('close')
  if (event.key === 'Tab') {
    const elements = [...closeButton.value.closest('.drawer').querySelectorAll('button, summary, a[href], input, select')].filter(e => e.getClientRects().length)
    const first=elements[0], last=elements.at(-1)
    if (event.shiftKey && document.activeElement===first) {event.preventDefault();last.focus()}
    else if (!event.shiftKey && document.activeElement===last) {event.preventDefault();first.focus()}
  }
}
onMounted(() => { previousFocus = document.activeElement; closeButton.value?.focus(); document.addEventListener('keydown', key) })
onBeforeUnmount(() => { document.removeEventListener('keydown', key); previousFocus?.focus() })
const list = v => { if (v == null || v === '') return []; if (Array.isArray(v)) return v; try { const parsed = JSON.parse(v); return Array.isArray(parsed) ? parsed : parsed == null ? [] : [parsed] } catch { return [v] } }
const value = (keys) => keys.map(k => props.record[k]).find(v => v != null && v !== '') ?? '未提供'
function eventTime(side,id,fallback) {
  for(const row of props.evidence?.event_rows || []) {
    if(side==='NPU' && Number(row['NPU Kernel Index'])===Number(id)) return row['NPU Duration(us)']
    const field=side==='GPU'?'GPU Kernel':'GPU Memory Activity', i=list(row[field+' Indices']).map(Number).indexOf(Number(id))
    if(side!=='NPU' && i>=0) return list(row[field+' Durations(us)'])[i] ?? fallback ?? '未提供'
  }
  return fallback ?? '未提供'
}
const groups = computed(() => [
  { side:'NPU',title:'NPU kernel', names:list(props.record['NPU Kernel Names'] ?? props.record['NPU Kernel Name'] ?? props.record.npu_kernel_name), ids:list(props.record['NPU Kernel Indices'] ?? props.record['NPU Kernel Index'] ?? props.record.npu_kernel_index), times:list(props.record['NPU Duration(us)'] ?? props.record.npu_duration_us), total:value(['NPU Total Time(us)','NPU Duration(us)','npu_duration_us']) },
  { side:'GPU',title:'GPU kernel', names:list(props.record['GPU Kernel Names']), ids:list(props.record['GPU Kernel Indices']), times:list(props.record['GPU Kernel Durations(us)']), total:value(['GPU Kernel Time(us)','GPU Total Time(us)']) },
  { side:'memory',title:'GPU 内存活动', names:list(props.record['GPU Memory Activity Names']), ids:list(props.record['GPU Memory Activity Indices']), times:list(props.record['GPU Memory Activity Durations(us)']), total:value(['GPU Memory Time(us)']) }
])
const info = computed(() => [
  ...(props.record['Manual Link ID'] ? [['关联来源','人工确认（非自动验证）'], ['确认理由',props.record['Manual Reason']]] : []),
  ['逻辑调用', value(['Whole OP','NPU CPU OP Name','cpu_op'])], ['GPU CPU op', value(['GPU OP Name'])],
  ['CPU 对齐',value(['CPU Alignment Status','cpu_alignment_status','CPU Match Confidence'])],
  ['Kernel 比较',value(['Kernel Comparison Status','kernel_comparison_status','Mapping Mode'])],
  ['对齐依据',value(['Alignment Basis','CPU Alignment Basis'])], ['待核实问题',value(['Issues','reason'])]
])
</script>
<template>
  <div class="drawer-backdrop" @click.self="$emit('close')"><section class="drawer" role="dialog" aria-modal="true" aria-label="调用与事件证据">
    <button ref="closeButton" class="close" @click="$emit('close')" aria-label="关闭详情">×</button><h2>调用与事件证据</h2><p class="muted">事件编号仅在本次运行内有效；未知耗时不会均摊或补零。</p>
    <dl class="evidence-info"><template v-for="[label,v] in info" :key="label"><dt>{{ label }}</dt><dd>{{ v }}</dd></template></dl>
    <p v-if="evidence?.has_more" class="notice">本次补充了前 {{ evidence.event_rows.length }} / {{ evidence.event_total }} 个 NPU 映射记录的事件耗时；其余未提供值保持未知，完整证据见 CSV。</p>
    <div class="event-groups"><section v-for="g in groups" :key="g.title"><h3>{{ g.title }} <small>合计 {{ g.total }} μs</small></h3><table><thead><tr><th>编号</th><th>名称</th><th>耗时 μs</th></tr></thead><tbody><tr v-for="(n,i) in g.names" :key="i"><td>{{ g.ids[i] ?? '—' }}</td><td class="kernel-name">{{ n }}</td><td>{{ eventTime(g.side,g.ids[i],g.times[i]) }}</td></tr><tr v-if="!g.names.length"><td colspan="3">此记录未提供事件列表</td></tr></tbody></table></section></div>
    <template v-if="evidence"><h3>诊断证据（{{ evidence.items.length }} / {{ evidence.total }} 条）</h3><article v-for="(e,i) in evidence.items" :key="i"><p>{{ e.cpu_alignment_status }} · {{ e.kernel_comparison_status }}</p><p>{{ e.reason }} {{ e.hypothesis }}</p>
      <table v-if="e.evidence?.npu_cpu"><thead><tr><th>CPU 元数据</th><th>NPU</th><th>GPU</th></tr></thead><tbody><tr v-for="field in ['name','input_shapes','input_dtypes']" :key="field" :class="{'metadata-difference':e.evidence.cpu_metadata_differences?.includes(field)}"><th>{{ field }}</th><td class="kernel-name">{{ JSON.stringify(e.evidence.npu_cpu?.[field]) ?? '未提供' }}</td><td class="kernel-name">{{ JSON.stringify(e.evidence.gpu_cpu?.[field]) ?? '未提供' }}</td></tr></tbody></table>
      <p v-if="e.evidence?.cpu_missing_metadata?.length">缺失或未展开：{{ e.evidence.cpu_missing_metadata.join(', ') }}，不等于真实冲突。</p><ol><li v-for="step in e.next_steps" :key="step">{{ step }}</li></ol>
      <details><summary>诊断原始字段</summary><pre>{{ JSON.stringify(e,null,2) }}</pre></details></article><p v-if="!evidence.total">当前生成阈值下没有逐 kernel 诊断，不代表没有差异。</p></template>
    <details><summary>原始数据 / 复制审计字段</summary><pre>{{ JSON.stringify(record,null,2) }}</pre></details>
  </section></div>
</template>
