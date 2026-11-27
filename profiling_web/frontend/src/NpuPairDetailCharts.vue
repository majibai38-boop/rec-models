<script setup>
import { computed, ref, watch } from 'vue'
import Chart from './Chart.vue'
import { PAIR_COLORS, colorOf, numelOf } from './npuPairView.js'
const props = defineProps({ job: Object, api: Function, scope: Object, thresholds: Object })
const rows = ref([]), total = ref(0), loading = ref(false), error = ref('')
const LIMIT = 200
const url = computed(() => '/jobs/' + props.job.id + '/npu-pair-table?' + new URLSearchParams({
  view: 'signatures', op: props.scope?.['OP Type'] || '', core: props.scope?.['Core Type'] || '',
  search: '', sort: 'candidate',
}))
// Primitive key, as in the tables: an equal-but-replaced job must not refetch.
watch(() => `${props.job.id}|${props.job.attempt}|${url.value}`, async (_, __, cleanup) => {
  let current = true
  cleanup(() => { current = false })
  rows.value = []; total.value = 0; error.value = ''; loading.value = true
  try {
    const result = await props.api(url.value + '&offset=0&limit=' + LIMIT)
    if (current) { rows.value = result.items || []; total.value = result.total || 0 }
  } catch (e) { if (current) error.value = e.message } finally { if (current) loading.value = false }
}, { immediate: true })

const fmt = value => value == null ? '—' : Number(value).toLocaleString('en-US', { maximumFractionDigits: 1 })
// One operator usually owns several shape groups with the same kernel name, so the shape is
// what tells them apart.
const shapeText = row => row['输入 Shape'] || ''
const title = row => [row['名称'], shapeText(row)].filter(Boolean).join(' · ') || row['OP Type'] || ''
const shortLabel = row => { const text = shapeText(row) || row['名称'] || ''; return text.length > 22 ? text.slice(0, 21) + '…' : text }
const peak = row => Math.max(row['对比耗时(us)'] || 0, row['基线耗时(us)'] || 0)

const top = computed(() => [...rows.value]
  .filter(row => row['对比耗时(us)'] != null || row['基线耗时(us)'] != null)
  .sort((a, b) => peak(b) - peak(a)).slice(0, 15).reverse())

const bars = computed(() => ({
  tooltip: {
    trigger: 'axis', confine: true, renderMode: 'richText',
    formatter: params => {
      const row = top.value[params?.[0]?.dataIndex]
      if (!row) return ''
      return `${title(row)}\n基线 ${fmt(row['基线耗时(us)'])} μs\n对比 ${fmt(row['对比耗时(us)'])} μs\n性能比 ${fmt(row['性能比(基线/对比)'])}`
    },
  },
  legend: { data: ['基线耗时', '对比耗时'], top: 0, left: 'center' },
  grid: { left: 8, right: 30, top: 46, bottom: 30, containLabel: true },
  xAxis: { type: 'value', name: 'μs' },
  yAxis: { type: 'category', data: top.value.map(shortLabel), axisLabel: { fontSize: 10 } },
  series: [
    { name: '基线耗时', type: 'bar', itemStyle: { color: '#7489ba' }, data: top.value.map(row => row['基线耗时(us)']) },
    { name: '对比耗时', type: 'bar', itemStyle: { color: '#167d78' }, data: top.value.map(row => row['对比耗时(us)']) },
  ],
}))

// The horizontal line is the threshold this type is measured against, when it has one.
const limit = computed(() => {
  const cv = props.scope?.CV, values = props.thresholds
  return cv === 'Cube' ? values?.cube : cv === 'Vector' ? values?.vector : null
})
const plotted = computed(() => rows.value
  .map(row => ({ row, x: numelOf(row['输入 Shape'], row['输入 dtype']) }))
  .filter(item => item.x != null && item.row['性能比(基线/对比)'] != null))

const scatter = computed(() => ({
  tooltip: {
    confine: true, renderMode: 'richText',
    formatter: p => `${p.data.name}\n元素数 ${fmt(p.value[0])}\n性能比 ${fmt(p.value[1])}\n对比耗时 ${fmt(p.data.time)} μs`,
  },
  grid: { left: 8, right: 30, top: 34, bottom: 40, containLabel: true },
  xAxis: { type: 'value', name: '首个输入元素数', nameLocation: 'middle', nameGap: 26, axisLabel: { fontSize: 10 } },
  yAxis: { type: 'value', name: '性能比', scale: true },
  series: [{
    type: 'scatter',
    markLine: {
      silent: true, symbol: 'none',
      data: [
        { yAxis: 1, lineStyle: { color: '#c3cfca', type: 'dashed' }, label: { formatter: '1.0', position: 'end', fontSize: 10 } },
        ...(Number(limit.value) > 0 ? [{ yAxis: limit.value, lineStyle: { color: '#c08a4a', type: 'dashed' },
          label: { formatter: '阈值', position: 'end', fontSize: 10 } }] : []),
      ],
    },
    data: plotted.value.map(({ row, x }) => {
      const ratio = row['性能比(基线/对比)']
      return {
        name: title(row), value: [x, ratio], time: row['对比耗时(us)'],
        symbolSize: Math.min(34, 6 + Math.sqrt(Math.max(row['对比耗时(us)'] || 0, 0))),
        itemStyle: { color: PAIR_COLORS[colorOf(ratio, row.CV, props.thresholds)] || PAIR_COLORS[''] },
      }
    }),
  }],
}))
</script>
<template>
  <p v-if="error" role="alert">{{ error }}</p>
  <p v-else-if="loading && !rows.length">正在准备图表…</p>
  <template v-else-if="rows.length">
    <div class="detail-charts">
      <section v-if="top.length"><h3>shape 耗时对比 · Top {{ top.length }}</h3><Chart :option="bars"/></section>
      <section v-if="plotted.length"><h3>输入规模与性能比</h3><Chart :option="scatter"/></section>
      <p v-else class="muted">此类型的 shape 不足以画输入规模图。</p>
    </div>
    <p class="muted small">按对比耗时取当前类型的前 {{ Math.min(total, 200) }} 个 shape 分组{{ total > 200 ? '（共 ' + total + ' 个）' : '' }}；气泡大小为对比耗时，虚线是该类型与性能比 1.0。图表只用于定位，不改变表格与比值口径。</p>
  </template>
</template>
<style scoped>
.detail-charts{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:14px 0 4px}
.detail-charts section{min-width:0}
.detail-charts h3{margin:0 0 6px;font-size:13px;color:#33413c;font-weight:600}
.detail-charts .chart{height:340px}
@media(max-width:1100px){.detail-charts{grid-template-columns:1fr}}
</style>
