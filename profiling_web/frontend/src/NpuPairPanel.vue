<script setup>
import { computed, reactive, watch } from 'vue'
const props = defineProps({ job: Object, busy: Boolean })
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
const columns = ['OP Type','状态','基线数量','对比数量','基线耗时(us)','对比耗时(us)','差值(us)','耗时比(对比/基线)','变化(%)','基线平均(us)','对比平均(us)']
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
    <h2>NPU 类型对比</h2>
    <p v-for="(label,side) in sides" :key="side">{{ label }}：{{ job.inputs[side+'_details'].name }} · Device {{ summary[side].device || '未标记' }} · {{ summary[side].step || '全量' }} · 排除无 Step 标记 {{ summary[side].excluded_unmarked }} 行</p>
    <p>总 kernel 耗时：{{ format(summary.overall['基线耗时(us)']) }} → {{ format(summary.overall['对比耗时(us)']) }} μs；对比 / 基线 = {{ format(summary.overall['耗时比(对比/基线)']) }}。</p>
    <p class="notice">比值大于 1 表示对比侧累计耗时更高。这是 kernel Duration 之和，多 stream 可重叠，不是 Step 墙钟时间。</p>
    <div style="overflow-x:auto"><table><thead><tr><th v-for="c in columns" :key="c">{{ c }}</th></tr></thead>
      <tbody><tr v-for="row in summary.types" :key="row['OP Type']"><td v-for="c in columns" :key="c">{{ ['OP Type','状态'].includes(c) ? row[c] : format(row[c]) }}</td></tr></tbody>
    </table></div>
    <p class="muted">按耗时差绝对值排序，展示 {{ summary.types.length }} / {{ summary.type_count }} 类。全部类型及 Shape / dtype 明细见下载。</p>
    <a v-for="file in summary.reports" :key="file" class="button" :href="'/api/jobs/'+job.id+'/artifacts/'+file">下载 {{ file }}</a>
  </section>
</template>
