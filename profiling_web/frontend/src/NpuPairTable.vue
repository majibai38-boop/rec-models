<script setup>
import { computed, ref, watch } from 'vue'
import {exportCsv} from './tableExport.js'
const props = defineProps({job:Object, api:Function, scope:Object})
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
  try {await exportCsv(props.api,url.value,props.job.id.slice(0,8)+(props.scope?'-shapes':'-types')+'.csv',
    {columns:columns.value,scope:{attempt:props.job.attempt,op:props.scope?.['OP Type'],core:props.scope?.['Core Type'],search:search.value,sort:sort.value}})}
  catch(e){error.value=e.message}finally{exporting.value=false}
}
const extra = ['输入 Shape','输入 dtype','输出 Shape','输出 dtype']
const columns = computed(()=>['OP Type','Core Type','CV',...(props.scope ? extra : []),'状态','基线数量','对比数量','基线耗时(us)','对比耗时(us)','差值(us)','耗时比(对比/基线)','变化(%)','基线平均(us)','对比平均(us)'])
const display = v => typeof v==='number' ? v.toLocaleString(undefined,{maximumFractionDigits:4}) : (v || '—')
watch(()=>[url.value,props.job.attempt,page.value,retry.value], async (_,__,cleanup)=>{
  let current = true
  cleanup(()=>{current=false})
  rows.value=[]; error.value=''; loading.value=true
  try {
    const result=await props.api(url.value+'&offset='+page.value*limit+'&limit='+limit)
    if(current){rows.value=result.items; total.value=result.total}
  } catch(e){if(current)error.value=e.message} finally {if(current)loading.value=false}
},{immediate:true})
</script>
<template>
  <form class="toolbar" @submit.prevent="filter">
    <input v-model="draft" :placeholder="scope?'搜索 Shape / dtype':'搜索 OP Type / Core Type'" aria-label="搜索当前表格" maxlength="200">
    <select v-model="sort" @change="filter" aria-label="排序"><option value="difference">耗时差绝对值降序</option><option value="baseline">基线耗时降序</option><option value="candidate">对比耗时降序</option><option value="ratio">耗时比降序</option></select>
    <button>筛选</button><button type="button" :disabled="exporting || loading || !total" @click="download">{{ exporting?'正在导出…':'导出当前筛选 CSV' }}</button>
  </form>
  <p v-if="error" role="alert">{{ error }} <button @click="retry++">重试</button></p>
  <p v-else-if="loading">正在读取比较结果…</p>
  <p v-else-if="!total" class="muted">没有符合当前筛选的记录。</p>
  <div v-else class="table-scroll"><table><thead><tr><th v-if="!scope">操作</th><th v-for="c in columns" :key="c">{{ c }}</th></tr></thead>
    <tbody><tr v-for="(r,i) in rows" :key="i"><td v-if="!scope"><button @click="$emit('details',r)">详细比较</button></td>
      <td v-for="c in columns" :key="c" :class="c==='耗时比(对比/基线)' ? r.ratio_color : ''">{{ display(r[c]) }}</td></tr></tbody></table></div>
  <div class="pagination"><span>共 {{ total }} 组 · 第 {{ page+1 }} 页</span><button :disabled="loading || !page" @click="page--">上一页</button><button :disabled="loading || (page+1)*limit>=total" @click="page++">下一页</button></div>
</template>
<style scoped>
td,th{white-space:nowrap}.green{background:#d9f2dc;color:#18562a;font-weight:600}.yellow{background:#fff2cc;color:#523f13}
</style>
