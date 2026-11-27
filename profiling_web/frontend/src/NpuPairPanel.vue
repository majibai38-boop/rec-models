<script setup>
import { computed, nextTick, reactive, ref, watch } from 'vue'
import NpuPairTable from './NpuPairTable.vue'
import NpuPairPipelineTable from './NpuPairPipelineTable.vue'
import ReportDownloads from './ReportDownloads.vue'
const props = defineProps({ job: Object, busy: Boolean, api:Function })
defineEmits(['run'])
const sides = {baseline:'基线 NPU', candidate:'对比 NPU'}
const opts = reactive({all_steps:false,baseline_device:'',candidate_device:'',baseline_step:'',candidate_step:''})
const devices = side => Object.keys(props.job.inspection?.[side]?.devices || {})
const steps = side => Object.keys(props.job.inspection?.[side]?.devices?.[opts[side+'_device']] || {}).filter(Boolean)
watch(()=>props.job.inspection, () => {
  for (const side of Object.keys(sides)) {
    const list = devices(side)
    if (list.length === 1) opts[side+'_device'] = list[0]
  }
}, {immediate:true})
const ready = computed(()=>validThresholds.value && Object.keys(sides).every(s=>devices(s).includes(opts[s+'_device']) &&
  (devices(s).length === 1 || opts[s+'_device']) &&
  (opts.all_steps || steps(s).includes(opts[s+'_step']))))
const payload = () => ({...opts, cube_threshold:Number(thresholds.cube), vector_threshold:Number(thresholds.vector),
  ...(opts.all_steps ? {baseline_step:'', candidate_step:''} : {})})
const summary = computed(()=>props.job.summary)
const format = v => v == null ? '—' : Number(v).toLocaleString(undefined,{maximumFractionDigits:4})
const selected=ref(null), showDetail=ref(false)
const detail=computed(()=>showDetail.value ? selected.value : null)
// Both views share one place: switching hides the other table (display:none, so its page and
// search survive) and brings the heading into view instead of leaving it above the fold.
const detailHeading = ref(null)
function scrollToHeading() {
  const top = detailHeading.value?.getBoundingClientRect().top
  if (top == null) return
  window.scrollTo({top: Math.max(top + window.scrollY - 88, 0), behavior: 'smooth'})
}
async function showDetailView(row) {
  if (row) selected.value = row
  if (!selected.value) return
  showDetail.value = true
  await nextTick()
  scrollToHeading()
}
async function hideDetailView() {
  showDetail.value = false
  await nextTick()
  scrollToHeading()
}
const classes = {Cube:'Cube',Vector:'Vector',Mixed:'混合核',Unknown:'未知'}
// Say which side fell back to the core type instead of a measured aicore_time counter.
const cvNote = computed(() => {
  const estimated = Object.entries(sides).filter(([key]) => summary.value?.[key]?.cv_source === 'core').map(([, label]) => label)
  if (!estimated.length) return 'CV 按 aicore_time 判定：正值为 Cube、零为 Vector，其余未知。'
  const detail = '按 Accelerator Core 估计（AI_CORE → Cube、AI_VECTOR_CORE → Vector、MIX → Mixed）'
  return estimated.length === 2 ? `两侧都没有 aicore_time 计数，Cube/Vector ${detail}。`
    : `${estimated.join('、')}没有 aicore_time 计数，该侧 Cube/Vector ${detail}。`
})
// Performance-ratio thresholds. They are sent with the run so the Excel/JSON use the same
// numbers the workbench colours by; later edits only recolour the live table.
const DEFAULTS = {cube:2, vector:1.1}
const stored = (() => {
  try {
    const v = JSON.parse(localStorage.getItem('profiling-pair-thresholds') || 'null')
    return v && Number(v.cube) > 0 && Number(v.vector) > 0 ? {cube:Number(v.cube), vector:Number(v.vector)} : null
  } catch { return null }
})()
const thresholds = reactive({...(stored || DEFAULTS)})
watch(thresholds, () => {
  try { localStorage.setItem('profiling-pair-thresholds', JSON.stringify({...thresholds})) } catch {}
})
// Adopt the job's thresholds once per job; re-fetching the same job must not overwrite edits.
let adoptedJob = null
watch(() => [props.job.id, summary.value?.thresholds], ([id, t]) => {
  if (id === adoptedJob || !t) return
  if (Number(t.cube) > 0 && Number(t.vector) > 0) {
    thresholds.cube = Number(t.cube); thresholds.vector = Number(t.vector); adoptedJob = id
  }
}, {immediate:true})
const validThresholds = computed(() => Number(thresholds.cube) > 0 && Number(thresholds.vector) > 0)
function resetThresholds() { thresholds.cube = DEFAULTS.cube; thresholds.vector = DEFAULTS.vector }
// Regenerated server-side with the thresholds currently on screen, so the file matches the table.
const belowHref = computed(() => {
  const cube = Number(thresholds.cube) > 0 ? Number(thresholds.cube) : DEFAULTS.cube
  const vector = Number(thresholds.vector) > 0 ? Number(thresholds.vector) : DEFAULTS.vector
  return `/api/jobs/${props.job.id}/npu-pair/below-threshold.xlsx?cube=${cube}&vector=${vector}`
})
const belowTitle = computed(() => `按当前阈值导出：Cube ${thresholds.cube} · Vector ${thresholds.vector}，含名称与输入输出 shape/dtype`)
</script>

<template>
  <section v-if="job.inspection && job.status !== 'succeeded'" class="panel">
    <h2>02 / 选择两个 NPU 的比较范围</h2>
    <div class="scope-grid">
      <div v-for="(label,side) in sides" :key="side">
        <h3>{{ label }}</h3>
        <label>Device ID<select v-model="opts[side+'_device']" @change="opts[side+'_step']=''">
          <option v-if="!devices(side).includes('')" disabled value="">选择设备</option>
          <option v-for="d in devices(side)" :key="d" :value="d" :disabled="!d && devices(side).length>1">{{ d || '未标记' }}</option>
        </select></label>
        <label>Step<select v-model="opts[side+'_step']" :disabled="opts.all_steps">
          <option value="">选择稳定 Step</option><option v-for="s in steps(side)" :key="s">{{ s }}</option>
        </select></label>
      </div>
    </div>
    <label class="check"><input type="checkbox" v-model="opts.all_steps">比较完整采集（包括预热和无 Step ID 的 kernel）</label>
    <div class="threshold-row">
      <span class="threshold-label">性能阈值</span>
      <label>Cube <input type="number" v-model.number="thresholds.cube" min="0.1" step="0.1" aria-label="Cube 性能比阈值"></label>
      <label>Vector <input type="number" v-model.number="thresholds.vector" min="0.1" step="0.1" aria-label="Vector 性能比阈值"></label>
      <span class="muted small">达标线，会写入本次的 Excel / JSON</span>
    </div>
    <p class="notice">没有 Step ID 时请选择完整采集。请核对两侧 batch、精度和工作量；设备编号不必相同。</p>
    <button class="primary" :disabled="busy || !ready" @click="$emit('run',payload())">开始 NPU 对比 →</button>
  </section>
  <section v-if="job.status==='succeeded' && summary" class="panel">
    <ReportDownloads :job="job"/>
    <h2>NPU 对比总览</h2>
    <p v-for="(label,side) in sides" :key="side">{{ label }}：{{ job.inputs[side+'_details'].name }} · Device {{ summary[side].device || '未标记' }} · {{ summary[side].step || '全量' }} · 排除无 Step 标记 {{ summary[side].excluded_unmarked }} 行</p>
    <div class="metrics"><article v-for="(label,side) in sides" :key="side"><span>{{ label }} · 总耗时</span>
      <strong>{{ format(summary.overall[side==='baseline'?'基线耗时(us)':'对比耗时(us)']) }} <small>μs</small></strong>
      <template v-if="summary[side].cv"><div class="cv-bar" aria-hidden="true"><i v-for="(g,k) in summary[side].cv" :key="k" :class="k" :style="{width:(g.time_pct||0)+'%'}"/></div>
        <p v-for="(g,k) in summary[side].cv" :key="k">{{ classes[k] }} · {{ format(g.time_us) }} μs · {{ g.time_pct==null?'—':format(g.time_pct)+'%' }}</p></template>
    </article><article><span>总体性能比 · 基线 / 对比</span><strong>{{ format(summary.overall['性能比(基线/对比)']) }}</strong><p>各端所选范围的全部 kernel</p></article></div>
    <p class="notice">性能比为耗时的反比：大于 1 表示对比侧用更少时间完成同样工作，小于 1 表示对比侧更慢。这是 kernel Duration 之和，多 stream 可重叠，不是 Step 墙钟时间。</p>
    <p class="muted">{{ cvNote }}占比分母为该侧全部 kernel 的 Duration，而非硬件利用率。</p>
    <template v-if="summary.conclusions?.length">
      <h2>结论</h2>
      <ul class="conclusions"><li v-for="(c,i) in summary.conclusions" :key="i"><strong>{{ c.kind }}</strong><span>{{ c.text }}</span></li></ul>
    </template>
    <template v-if="summary.pipeline">
      <h2>NPU 流水初筛 · 按算子</h2>
      <p class="muted">按每个 kernel 的 PipeUtilization 归类后汇总到 OP Type + Core Type；类别耗时互斥，无法分类的时间一并计入 Unknown 分母。仅为瓶颈候选，不是 Roofline 饱和结论。</p>
      <NpuPairPipelineTable :job="job" :api="api"/>
    </template>
    <template v-if="summary.schema_version>=2">
      <div class="tabs" role="tablist" aria-label="NPU 比较视图">
        <button role="tab" :aria-selected="!detail" :class="{active:!detail}" @click="hideDetailView">类型对比</button>
        <button role="tab" :aria-selected="!!detail" :disabled="!selected" :class="{active:!!detail}" @click="showDetailView()">算子详情</button>
      </div>
      <div class="threshold-row">
        <span class="threshold-label">性能阈值</span>
        <label>Cube <input type="number" v-model.number="thresholds.cube" min="0.1" step="0.1" aria-label="Cube 性能比阈值"></label>
        <label>Vector <input type="number" v-model.number="thresholds.vector" min="0.1" step="0.1" aria-label="Vector 性能比阈值"></label>
        <button type="button" @click="resetThresholds">恢复默认</button>
        <a class="button below-export" :href="belowHref" :title="belowTitle">↓ 导出未达标算子</a>
        <span class="ratio-legend"><i class="dark-green">≥ 1.5× 阈值</i><i class="green">≥ 阈值</i><i class="yellow">≥ 0.5× 阈值</i><i class="dark-yellow">&lt; 0.5× 阈值</i></span>
      </div>
      <p class="muted small">阈值为性能比（基线 / 对比）的达标线：绿色代表达到或超过，黄色代表未达标；Mixed / Unknown 不套用阈值。表内配色随输入实时计算，Excel 使用分析时的阈值。</p>
      <h2 ref="detailHeading">{{ detail ? '详细比较 · '+detail['OP Type']+' / '+(detail['Core Type']||'未标记') : '类型对比' }}</h2>
      <button v-if="detail" @click="hideDetailView">← 返回类型对比</button>
      <NpuPairTable v-show="!detail" :job="job" :api="api" :thresholds="thresholds" @details="showDetailView($event)"/>
      <NpuPairTable v-if="detail" :job="job" :api="api" :scope="detail" :thresholds="thresholds"/>
    </template>
    <template v-else><h2>类型对比（历史结果）</h2><p class="notice">此结果没有 CV 和详细比较索引，可复用输入重新分析。</p>
      <div class="table-scroll"><table><thead><tr><th>类型</th><th>基线 μs</th><th>对比 μs</th><th>对比 / 基线</th></tr></thead><tbody>
      <tr v-for="r in summary.types" :key="r['OP Type']"><td>{{ r['OP Type'] }}</td><td>{{ format(r['基线耗时(us)']) }}</td><td>{{ format(r['对比耗时(us)']) }}</td><td>{{ format(r['耗时比(对比/基线)']) }}</td></tr></tbody></table></div></template>
  </section>
</template>
<style scoped>
.cv-bar{display:flex;height:8px;background:#edf1ef;border-radius:4px;overflow:hidden;margin:12px 0}.Cube{background:#167d78}.Vector{background:#7489ba}.Mixed{background:#c08a4a}.Unknown{background:#8c9198}
.conclusions{list-style:none;padding:0;margin:12px 0;display:grid;gap:10px}.conclusions li{display:flex;gap:12px;align-items:flex-start;line-height:1.65}.conclusions strong{flex:0 0 76px;color:#167d78}
.threshold-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:16px 0 8px;padding:12px 16px;background:#f7faf9;border:1px solid #e3ece7;border-radius:9px}
.threshold-label{font-size:12px;font-weight:600;color:#2f6d59}
.threshold-row label{display:flex;align-items:center;gap:7px;font-size:12px;color:#617569}
.threshold-row input{width:82px;padding:6px 9px}
.threshold-row button{font-size:11px;padding:6px 10px}
.below-export{font-size:11px;padding:6px 11px;border-style:dashed;color:#2f6d59}
.ratio-legend{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.ratio-legend i{font-style:normal;font-size:10px;padding:3px 8px;border-radius:4px;white-space:nowrap}
.ratio-legend .dark-green{background:#197746;color:#fff}
.ratio-legend .green{background:#dcefd9;color:#26533a}
.ratio-legend .yellow{background:#fff0b5;color:#6f5a12}
.ratio-legend .dark-yellow{background:#ffd966;color:#6a4a00}
</style>
