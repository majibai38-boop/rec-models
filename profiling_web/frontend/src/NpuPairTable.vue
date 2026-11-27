<script setup>
import { computed, ref, watch } from 'vue'
import { exportCsv } from './tableExport.js'
import { colorOf } from './npuPairView.js'
const props = defineProps({job:Object, api:Function, scope:Object, thresholds:Object, focus:String})
defineEmits(['details'])
const page = ref(0), total = ref(0), rows = ref([]), error = ref(''), loading = ref(false), retry = ref(0)
const limit = 30
const search=ref(''), draft=ref(''), sort=ref('difference'), exporting=ref(false)
const url=computed(()=>{
  const params=new URLSearchParams({view:props.scope?'signatures':'types',search:search.value,sort:sort.value})
  if(props.scope){params.set('op',props.scope['OP Type']);params.set('core',props.scope['Core Type'])}
  return '/jobs/'+props.job.id+'/npu-pair-table?'+params
})
function filter(){search.value=draft.value.trim(); page.value=0}
async function download(){
  exporting.value=true;error.value=''
  // 流水 is a two-bar cell, not a value; its numbers live in the workbook and JSON.
  const fields = columns.value.filter(c => c !== FLOW)
  try {await exportCsv(props.api,url.value,props.job.id.slice(0,8)+(props.scope?'-shapes':'-types')+'.csv',
    {columns:fields,scope:{attempt:props.job.attempt,op:props.scope?.['OP Type'],core:props.scope?.['Core Type'],search:search.value,sort:sort.value}})}
  catch(e){error.value=e.message}finally{exporting.value=false}
}
const extra = ['名称','输入 Shape','输入 dtype','输出 Shape','输出 dtype']
// One column for both sides, as the GPU/NPU type table shows the NPU pipeline in one column.
const FLOW = '流水'
const flowText = data => (data?.categories || []).map(c => `${c.label} ${c.time_pct == null ? '—' : c.time_pct.toFixed(1) + '%'}`).join('，')
const flowTitle = r => `基线：${flowText(r['基线流水']) || '未采集'}\n对比：${flowText(r['对比流水']) || '未采集'}`
// Recoloured in the browser so the thresholds can be edited without re-running the analysis.
const cellColor = r => colorOf(r['性能比(基线/对比)'], r.CV, props.thresholds)
const columns = computed(()=>['OP Type','Core Type','CV',...(props.scope ? extra : []),'状态','基线数量','对比数量','基线耗时(us)','对比耗时(us)','性能比(基线/对比)',...(props.scope ? [] : [FLOW])])
const display = v => typeof v==='number' ? v.toLocaleString(undefined,{maximumFractionDigits:4}) : (v || '—')
// Key the fetch on primitives only: replacing the parent job object with an equal one must
// not trigger a refetch, which would empty the table and move the page scroll.
let lastJob = ''
watch(()=>`${props.job.id}|${props.job.attempt}|${page.value}|${retry.value}|${url.value}`, async (_,__,cleanup)=>{
  let current = true
  cleanup(()=>{current=false})
  if (props.job.id !== lastJob) { rows.value=[]; total.value=0; lastJob=props.job.id }  // never show another job's rows
  error.value=''; loading.value=true
  try {
    const result=await props.api(url.value+'&offset='+page.value*limit+'&limit='+limit)
    if(current){rows.value=result.items; total.value=result.total}
  } catch(e){if(current)error.value=e.message} finally {if(current)loading.value=false}
}, {immediate:true})
// Charts hand over one shape group to focus; applying it as the search keeps the table honest
// (the filter box shows what is filtered) and the user can clear it again.
watch(()=>props.focus, value=>{
  if (value == null || value === search.value) return
  draft.value = value; search.value = value; page.value = 0
})
</script>
<template>
  <div class="pair-table">
  <form class="toolbar" @submit.prevent="filter">
    <input v-model="draft" :placeholder="scope?'搜索 Shape / dtype':'搜索 OP Type / Core Type'" aria-label="搜索当前表格" maxlength="200">
    <select v-model="sort" @change="filter" aria-label="排序"><option value="difference">耗时差绝对值降序</option><option value="baseline">基线耗时降序</option><option value="candidate">对比耗时降序</option><option value="ratio">性能比降序</option></select>
    <button>筛选</button><button type="button" :disabled="exporting || loading || !total" @click="download">{{ exporting?'正在导出…':'导出当前筛选 CSV' }}</button>
  </form>
  <p v-if="error" role="alert">{{ error }} <button @click="retry++">重试</button></p>
  <p v-else-if="loading && !rows.length">正在读取比较结果…</p>
  <p v-else-if="!total" class="muted">没有符合当前筛选的记录。</p>
  <div v-else class="table-scroll"><table><thead><tr><th v-if="!scope">操作</th><th v-for="c in columns" :key="c">{{ c }}</th></tr></thead>
    <tbody><tr v-for="(r,i) in rows" :key="i"><td v-if="!scope"><button @click="$emit('details',r)">详细比较</button></td>
      <td v-for="c in columns" :key="c" :class="c==='性能比(基线/对比)' ? cellColor(r) : (c===FLOW ? 'flow-cell' : '')" :title="c===FLOW ? flowTitle(r) : null">
        <template v-if="c===FLOW">
          <div v-for="side in [['基','基线流水'],['对','对比流水']]" :key="side[1]" class="flow-row">
            <small>{{ side[0] }}</small>
            <span v-if="r[side[1]]?.categories?.length" class="flow-bar">
              <i v-for="seg in r[side[1]].categories" :key="seg.category" :class="'pipe-'+seg.category" :style="{width:(seg.time_pct||0)+'%'}"/>
            </span>
            <span v-else class="muted">—</span>
          </div>
        </template>
        <template v-else>{{ display(r[c]) }}</template>
      </td></tr></tbody></table></div>
  <div class="pagination"><span>共 {{ total }} 组 · 第 {{ page+1 }} 页</span><button :disabled="loading || !page" @click="page--">上一页</button><button :disabled="loading || (page+1)*limit>=total" @click="page++">下一页</button></div>
  </div>
</template>
<style scoped>
td,th{white-space:nowrap}
.flow-cell{min-width:158px}
.flow-row{display:flex;align-items:center;gap:5px;line-height:14px}
.flow-row small{width:11px;color:#61756e;font-size:9px}
.flow-bar{display:inline-flex;width:126px;height:8px;background:#edf0f2;border-radius:3px;overflow:hidden}
.flow-bar i{height:100%}
.pipe-compute{background:#487cbd}.pipe-memory{background:#249787}.pipe-scalar{background:#9972b5}
.pipe-mixed{background:#cb8a3d}.pipe-no_bound{background:#b5a54f}.pipe-unknown{background:#969da8}
.dark-green{background:#197746;color:#fff;font-weight:600}
.green{background:#dcefd9;color:#26533a;font-weight:600}
.yellow{background:#fff0b5;color:#6f5a12}
.dark-yellow{background:#ffd966;color:#6a4a00;font-weight:600}
</style>
