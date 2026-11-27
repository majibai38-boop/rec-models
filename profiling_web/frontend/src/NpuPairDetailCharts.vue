<script setup>
import { computed, ref, watch } from 'vue'
import Chart from './Chart.vue'
import { PAIR_COLORS, colorOf, inputCountOf, numelOf } from './npuPairView.js'
const props = defineProps({ job: Object, api: Function, scope: Object, thresholds: Object })
const emit = defineEmits(['select'])
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
    { name: '基线耗时', type: 'bar', itemStyle: { color: '#7489ba' },
      data: top.value.map(row => ({ value: row['基线耗时(us)'], payload: row })) },
    { name: '对比耗时', type: 'bar', itemStyle: { color: '#167d78' },
      data: top.value.map(row => ({ value: row['对比耗时(us)'], payload: row })) },
  ],
}))

// Scatter X axis follows the selected input tensor, as in the GPU/NPU workbench.
const inputIndex = ref(0)
const inputCount = computed(() => inputCountOf(rows.value))
watch(inputCount, total => { if (inputIndex.value >= total) inputIndex.value = 0 })

// The Y axis can show the ratio or either side's duration; the colours keep following the ratio.
const Y_METRICS = {
  ratio: { label: '性能比', field: '性能比(基线/对比)', axis: '性能比' },
  candidate: { label: '对比耗时', field: '对比耗时(us)', axis: 'μs' },
  baseline: { label: '基线耗时', field: '基线耗时(us)', axis: 'μs' },
}
const yMetric = ref('ratio')
const metric = computed(() => Y_METRICS[yMetric.value] || Y_METRICS.ratio)

// The horizontal line is the threshold this type is measured against, when it has one.
const limit = computed(() => {
  const cv = props.scope?.CV, values = props.thresholds
  return cv === 'Cube' ? values?.cube : cv === 'Vector' ? values?.vector : null
})
const plotted = computed(() => rows.value
  .map(row => ({ row, x: numelOf(row['输入 Shape'], row['输入 dtype'], inputIndex.value) }))
  .filter(item => item.x != null && item.row[metric.value.field] != null))

const scatter = computed(() => ({
  tooltip: {
    confine: true, renderMode: 'richText',
    formatter: p => `${p.data.name}\ninput[${inputIndex.value}] 元素数 ${fmt(p.value[0])}\n性能比 ${fmt(p.data.ratio)}\n对比耗时 ${fmt(p.data.time)} μs\n基线耗时 ${fmt(p.data.base)} μs`,
  },
  grid: { left: 8, right: 30, top: 34, bottom: 40, containLabel: true },
  xAxis: { type: 'value', name: `input[${inputIndex.value}] 元素数`, nameLocation: 'middle', nameGap: 26, axisLabel: { fontSize: 10 } },
  yAxis: { type: 'value', name: metric.value.axis, scale: true },
  series: [{
    type: 'scatter',
    markLine: yMetric.value === 'ratio' ? {
      silent: true, symbol: 'none',
      data: [
        { yAxis: 1, lineStyle: { color: '#c3cfca', type: 'dashed' }, label: { formatter: '1.0', position: 'end', fontSize: 10 } },
        ...(Number(limit.value) > 0 ? [{ yAxis: limit.value, lineStyle: { color: '#c08a4a', type: 'dashed' },
          label: { formatter: '阈值', position: 'end', fontSize: 10 } }] : []),
      ],
    } : { data: [] },
    data: plotted.value.map(({ row, x }) => {
      const ratio = row['性能比(基线/对比)']
      return {
        name: title(row), value: [x, row[metric.value.field]], payload: row,
        ratio, time: row['对比耗时(us)'], base: row['基线耗时(us)'],
        symbolSize: Math.min(34, 6 + Math.sqrt(Math.max(row['对比耗时(us)'] || 0, 0))),
        itemStyle: { color: PAIR_COLORS[colorOf(ratio, row.CV, props.thresholds)] || PAIR_COLORS[''] },
      }
    }),
  }],
}))

// Clicking a bar or a dot filters the table below to that shape group.
function pick(params) { const row = params?.data?.payload; if (row) emit('select', row) }
</script>
<template>
  <p v-if="error" role="alert">{{ error }}</p>
  <p v-else-if="loading && !rows.length">正在准备图表…</p>
  <template v-else-if="rows.length">
    <div class="detail-charts">
      <section v-if="top.length">
        <h3>shape 耗时对比 · Top {{ top.length }}</h3>
        <Chart :option="bars" @select="pick"/>
      </section>
      <section v-if="plotted.length">
        <div class="chart-head">
          <h3>输入规模与{{ metric.label }}</h3>
          <div class="chart-controls">
            <label class="chart-input">Y 轴
              <select v-model="yMetric" aria-label="散点图 Y 轴">
                <option value="ratio">性能比</option>
                <option value="candidate">对比耗时</option>
                <option value="baseline">基线耗时</option>
              </select>
            </label>
            <label v-if="inputCount > 1" class="chart-input">输入位置
              <select v-model.number="inputIndex" aria-label="散点图输入位置">
                <option v-for="n in inputCount" :key="n" :value="n - 1">input[{{ n - 1 }}]</option>
              </select>
            </label>
          </div>
        </div>
        <Chart :option="scatter" @select="pick"/>
      </section>
      <p v-else class="muted">此类型的 shape 不足以画输入规模图。</p>
    </div>
    <p class="muted small">按对比耗时取当前类型的前 {{ Math.min(total, 200) }} 个 shape 分组{{ total > 200 ? '（共 ' + total + ' 个）' : '' }}；气泡大小为对比耗时{{ yMetric === 'ratio' ? '，虚线是该类型的性能阈值与 1.0' : '' }}。点柱子或气泡会把下方表格筛到该 shape，便于逐项查证。</p>
  </template>
</template>
<style scoped>
.detail-charts{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:14px 0 4px}
.detail-charts section{min-width:0}
.detail-charts h3{margin:0 0 6px;font-size:13px;color:#33413c;font-weight:600}
.detail-charts .chart{height:340px}
.chart-head{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
.chart-controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.chart-input{display:flex;align-items:center;gap:6px;font-size:11px;color:#61756e;margin-bottom:6px}
.chart-input select{font-size:11px;padding:4px 7px}
@media(max-width:1100px){.detail-charts{grid-template-columns:1fr}}
</style>
