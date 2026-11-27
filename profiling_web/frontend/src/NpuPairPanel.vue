<script setup>
import { computed, reactive, ref, watch } from 'vue'
import NpuPairTable from './NpuPairTable.vue'
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
const ready = computed(()=>Object.keys(sides).every(s=>devices(s).includes(opts[s+'_device']) &&
  (devices(s).length === 1 || opts[s+'_device']) &&
  (opts.all_steps || steps(s).includes(opts[s+'_step']))))
const payload = () => ({...opts, ...(opts.all_steps ? {baseline_step:'', candidate_step:''} : {})})
const summary = computed(()=>props.job.summary)
const format = v => v == null ? '—' : Number(v).toLocaleString(undefined,{maximumFractionDigits:4})
const selected=ref(null), showDetail=ref(false)
const detail=computed(()=>showDetail.value ? selected.value : null)
const classes = {Cube:'Cube',Vector:'Vector',Unknown:'未知'}
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
    </article><article><span>总体耗时比 · 对比 / 基线</span><strong>{{ format(summary.overall['耗时比(对比/基线)']) }}</strong><p>各端所选范围的全部 kernel</p></article></div>
    <p class="notice">比值大于 1 表示对比侧累计耗时更高。这是 kernel Duration 之和，多 stream 可重叠，不是 Step 墙钟时间。</p>
    <p class="muted">CV 按 aicore_time 分类：正值为 Cube、零为 Vector，其余未知。占比分母为该侧全部 kernel 的 Duration，而非硬件利用率。</p>
    <template v-if="summary.schema_version>=2">
      <div class="tabs" role="tablist" aria-label="NPU 比较视图">
        <button role="tab" :aria-selected="!detail" :class="{active:!detail}" @click="showDetail=false">类型对比</button>
        <button role="tab" :aria-selected="!!detail" :disabled="!selected" :class="{active:!!detail}" @click="showDetail=true">算子详情</button>
      </div>
      <h2>{{ detail ? '详细比较 · '+detail['OP Type']+' / '+(detail['Core Type']||'未标记') : '类型对比' }}</h2>
      <p>绿色：Cube 比值 &gt; 2.0 或 Vector 比值 &gt; 1.1；其他有效比值为浅黄色。无比值不着色。</p>
      <button v-if="detail" @click="showDetail=false">← 返回类型对比</button>
      <NpuPairTable v-show="!detail" :job="job" :api="api" @details="selected=$event; showDetail=true"/>
      <NpuPairTable v-if="detail" :job="job" :api="api" :scope="detail"/>
    </template>
    <template v-else><h2>类型对比（历史结果）</h2><p class="notice">此结果没有 CV 和详细比较索引，可复用输入重新分析。</p>
      <div class="table-scroll"><table><thead><tr><th>类型</th><th>基线 μs</th><th>对比 μs</th><th>对比 / 基线</th></tr></thead><tbody>
      <tr v-for="r in summary.types" :key="r['OP Type']"><td>{{ r['OP Type'] }}</td><td>{{ format(r['基线耗时(us)']) }}</td><td>{{ format(r['对比耗时(us)']) }}</td><td>{{ format(r['耗时比(对比/基线)']) }}</td></tr></tbody></table></div></template>
  </section>
</template>
<style scoped>
.cv-bar{display:flex;height:8px;background:#edf1ef;border-radius:4px;overflow:hidden;margin:12px 0}.Cube{background:#167d78}.Vector{background:#7489ba}.Unknown{background:#8c9198}
</style>
