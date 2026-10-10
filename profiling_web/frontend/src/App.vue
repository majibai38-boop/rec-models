<script setup>
import { computed, onMounted, onBeforeUnmount, ref, reactive, watch, nextTick } from 'vue'
import Chart from './Chart.vue'
import WholeCallTable from './WholeCallTable.vue'
import TypeTable from './TypeTable.vue'
import EvidenceDrawer from './EvidenceDrawer.vue'
import ManualLinkDialog from './ManualLinkDialog.vue'
import CoreBreakdown from './CoreBreakdown.vue'
import { cleanView, readLocation, viewQuery, readSavedViews, csvCell, normalizeStep } from './viewState.js'
const logged = ref(false), token = ref(''), error = ref(''), busy = ref(false), config = ref({})
const jobs = ref([]), job = ref(null), tab = ref('overview'), name = ref(''), files = reactive({}), uploading = ref(''), uploadPercent = ref(0)
const callFilter = ref({})
const callSearch = ref(''), callSort = ref('npu_time'), callStatus = ref(''), detailed = ref(false), focus = ref('')
const jobSearch = ref(''), jobStatus = ref(''), jobArchived = ref(false), jobPage = ref(0), jobTotal = ref(0)
const editingName = ref(false), editedName = ref(''), service = ref(null), loadingResults = ref(false), loadingRecords = ref(false)
const savedViews = ref(readSavedViews(localStorage.getItem('profiling-views-v1'))), savedViewIndex = ref(''), viewName = ref(''), savingView = ref(false)
const trail = ref([]), chartCache = new Map()
const manualType = ref(null)
function manualSubmitted(next) {job.value=next;manualType.value=null;chartCache.clear();action(refresh)}
function manualUpgrade() {manualType.value=null;cloneJob()}
let restoring = false, lastRefresh = 0
const snapshot = () => cleanView({tab:tab.value,search:search.value,sort:sort.value,hotspot:hotspot.value,page:page.value,recordPage:recordPage.value,recordKind:recordKind.value,scope:scope.value,callFilter:callFilter.value,callSearch:callSearch.value,callSort:callSort.value,callStatus:callStatus.value,detailed:detailed.value,focus:focus.value})
function applyView(v) {
  const x=cleanView(v); tab.value=x.tab; search.value=x.search; sort.value=x.sort; hotspot.value=x.hotspot; page.value=x.page; recordPage.value=x.recordPage; recordKind.value=x.recordKind
  scope.value=Object.keys(x.scope).length ? x.scope : null; callFilter.value=x.callFilter; callSearch.value=x.callSearch; callSort.value=x.callSort; callStatus.value=x.callStatus; detailed.value=x.detailed; focus.value=x.focus
}
function writeLocation(push=false) { if (!restoring && job.value) history[push?'pushState':'replaceState']({},'', '?' + viewQuery(job.value.id,snapshot())) }
function remember(label) { trail.value.push({label,view:snapshot()}); focus.value='' }
async function backTo(index) { const item=trail.value[index]; trail.value=trail.value.slice(0,index); applyView(item.view); detail.value=null; writeLocation(true); await action(loadResults) }
function saveView() { const label=viewName.value; if (!label?.trim()) return; const view={...snapshot(), page:0, recordPage:0, scope:{}, callFilter:{}, focus:''}; savedViews.value=[...savedViews.value.filter(v=>v.name!==label.trim()),{name:label.trim().slice(0,60),view}].slice(-20); localStorage.setItem('profiling-views-v1',JSON.stringify(savedViews.value)); savingView.value=false; viewName.value=''; savedViewIndex.value=String(savedViews.value.length-1) }
async function useView() { if(savedViewIndex.value==='') return; const v=savedViews.value[Number(savedViewIndex.value)]; if (!v) return; remember('上一个视图'); applyView(v.view); writeLocation(true); await action(loadResults) }
function deleteView() { if(savedViewIndex.value==='') return; savedViews.value.splice(Number(savedViewIndex.value),1); localStorage.setItem('profiling-views-v1',JSON.stringify(savedViews.value)); savedViewIndex.value='' }
async function renameJob() { await action(async()=>{ job.value=await api('/jobs/'+job.value.id,'PATCH',{name:editedName.value.trim()}); editingName.value=false; await refresh() }) }
async function archiveJob() { await action(async()=>{ job.value=await api('/jobs/'+job.value.id,'PATCH',{archived:!job.value.archived}); await refresh() }) }
async function cloneJob() { await action(async()=>{ const j=await api('/jobs/'+job.value.id+'/clone','POST'); jobArchived.value=false; jobSearch.value=''; jobPage.value=0; await refresh(); await select(j.id) }) }
async function showService() { await action(async()=>{service.value=service.value ? null : await api('/health')}) }
async function findJobs() {jobPage.value=0;await action(refresh)}
async function clearFilters() {search.value='';hotspot.value=false;scope.value=null;page.value=0;recordPage.value=0;await action(loadResults)}
function closeDetail() {detail.value=null;evidence.value=null;focus.value='';writeLocation()}
async function restoreLocation() {
  const target=readLocation(location.search); const id=target.job || localStorage.getItem('profiling-job')
  if (!id) return
  restoring=true
  try {await select(id,target.view);await nextTick()} finally {restoring=false}
}
function onPopState() {action(restoreLocation)}
function recordParams() {
  const kind=tab.value==='whole'?'whole_calls':tab.value==='calls'?'call_details':tab.value==='distribution'?'gpu_unmapped':recordKind.value
  const calls=['whole','calls'].includes(tab.value)
  return {kind,...(calls?callFilter.value:tab.value==='coverage'?(scope.value||{}):{}),
    ...(calls?{search:callSearch.value,sort:callSort.value,...(callStatus.value?{comparison_status:callStatus.value}:{})}:tab.value==='coverage'?{search:search.value,...(hotspot.value?{coverage_max:90,npu_ratio_min:1}:{})}:{})}
}
const supportsWhole = computed(()=>summary.value?.reports?.includes('whole_call_mapping.csv'))
async function exportView() { await action(async()=>{
  const id=job.value.id, state=snapshot(), params=['overview','coverage'].includes(tab.value)?{search:search.value,sort:sort.value,...(tab.value==='coverage'?(scope.value||{}):{}),...(hotspot.value?{coverage_max:90,npu_ratio_min:1}:{})}:recordParams()
  let result=[], offset=0
  while(true) {const data=await api('/jobs/'+id+'/results?'+new URLSearchParams({...params,offset,limit:200})); if(data.total>100000) throw Error('筛选结果超过 10 万行，请缩小范围或下载完整报告'); result.push(...data.items); if(!data.has_more) break;offset+=200}
  const cell=csvCell
  const fields=[...new Set(result.flatMap(r=>Object.keys(r)))], lines=[['Export scope',JSON.stringify({job:id,view:state})],fields,...result.map(r=>fields.map(k=>r[k]))]
  const url=URL.createObjectURL(new Blob(['\uFEFF'+lines.map(row=>row.map(cell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=id.slice(0,8)+'-'+tab.value+'-filtered.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)
}) }
const rows = ref([]), chartRows = ref([]), total = ref(0), page = ref(0), search = ref(''), sort = ref('npu_time'), hotspot = ref(false)
const detail = ref(null), evidence = ref(null), scope = ref(null), records = ref([]), recordTotal = ref(0), recordPage = ref(0), recordKind = ref('reasons')
const options = reactive({ gpu_pid: '', npu_pid: '', gpu_step: '', npu_step: '', all_steps: false, diagnostic_coverage_threshold: 90, include_communication: false })
const unassignedFiles=ref([])
let uploadRequest=null
function acceptFiles(event) {
  if(busy.value)return
  const incoming=[...(event.dataTransfer?.files || event.target.files || [])]
  for(const file of incoming) {
    const n=file.name.toLowerCase()
    const role=n.endsWith('.csv')&&n.includes('kernel_details')?'npu_kernel_details':n.endsWith('.json')&&n.includes('gpu')?'gpu_trace':n.endsWith('.json')&&(n.includes('npu')||n.includes('trace_view'))?'npu_trace':null
    if(role&&!files[role]) files[role]=file
    else unassignedFiles.value.push(file)
  }
}
function assignFile(index,role) {if(!role)return;files[role]=unassignedFiles.value[index];unassignedFiles.value.splice(index,1)}
function cancelUpload() {uploadRequest?.abort()}
const errorAdvice=computed(()=>{
 const e=job.value?.error||''
 return e.includes('内存')?'缩小采集范围或提高内存限制，然后重试。':e.includes('超时')||e.includes('运行时间')?'缩小输入或提高超时限制，然后重试。':e.includes('存储')||e.includes('配额')?'在任务列表删除不再需要的数据，再重试。':e.includes('输入')||e.includes('预检查')?'检查预检查提示；输入错误时新建任务上传修正文件。':'检查输入和比较范围。重试会保留旧运行目录；原始错误仅保存在服务器日志。'
})
const roles = [{ key: 'gpu_trace', title: 'GPU trace', hint: 'PyTorch profiler · JSON', accept: '.json' }, { key: 'npu_trace', title: 'NPU trace', hint: 'Ascend trace_view · JSON', accept: '.json' }, { key: 'npu_kernel_details', title: 'NPU kernel details', hint: 'kernel_details · CSV', accept: '.csv' }]
const labels = { uploading: '等待上传', queued: '排队中', running: '分析中', needs_selection: '等待确认范围', succeeded: '分析完成', failed: '失败', cancelled: '已取消', interrupted: '已中断', inspection: '输入预检查', gpu_parse: '解析 GPU trace', npu_parse: '解析 NPU trace', detail_link: '关联 NPU 明细', matching: '匹配 CPU / kernel', reports: '生成报告', validation: '校验计时与映射', indexing: '构建查询索引', starting: '启动分析进程' }
const tabs = [['overview', '类型对比'], ['whole', '整体对比'], ['coverage', '覆盖率诊断'], ['calls', '调用明细'], ['distribution', '耗时分布']]
const summary = computed(() => job.value?.summary)
const overall = computed(() => summary.value?.overall_statistics?.[0])
const active = computed(() => ['running', 'queued'].includes(job.value?.status))
const selectable = computed(() => job.value?.inspection && ['needs_selection', 'failed', 'cancelled', 'interrupted'].includes(job.value?.status))
const elapsed = ref(0)
let timer, loading = false, selectionSequence = 0, resultsSequence = 0, recordsSequence = 0
const fmt = (v, n = 2) => v == null || v === '' ? '—' : Number(v).toLocaleString('en-US', { maximumFractionDigits: n })
const ratioClass = v => v == null || v === '' ? '' : v >= 2 ? 'ratio-dark' : v >= 1 ? 'ratio-light' : 'ratio-yellow'
const filesize = v => `${fmt(v / 1024 ** 2, 1)} MB`
async function api(path, method = 'GET', body) {
  const response = await fetch('/api' + path, { method, headers: body === undefined ? {} : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
  if (response.status === 401) logged.value = false
  if (!response.ok) {
    const e = await response.json().catch(() => ({}))
    const failure = new Error(typeof e.detail === 'string' ? e.detail : e.detail?.message || JSON.stringify(e.detail || response.statusText))
    failure.detail = e.detail
    throw failure
  }
  return response.json()
}
async function action(fn) { error.value = ''; busy.value = true; try { await fn() } catch (e) { error.value = e.message } finally { busy.value = false } }
async function login() { await action(async () => { await api('/session', 'POST', { token: token.value }); token.value = ''; logged.value = true; await refresh(); await restoreLocation() }) }
async function logout() { await action(async () => { await api('/session', 'DELETE'); logged.value = false; job.value = null }) }
async function refresh() { const [cfg,data]=await Promise.all([api('/config'),api('/jobs?'+new URLSearchParams({offset:jobPage.value*30,limit:30,search:jobSearch.value,status:jobStatus.value,archived:jobArchived.value}))]);config.value=cfg;jobs.value=data.items;jobTotal.value=data.total;lastRefresh=Date.now() }
async function select(id, view=null) {
  manualType.value=null
  const sequence = ++selectionSequence
  const next = await api('/jobs/' + id)
  if (sequence !== selectionSequence) return
  job.value = next; localStorage.setItem('profiling-job', id); page.value = 0; recordPage.value = 0; scope.value = null; detail.value = null; evidence.value = null; records.value = []; rows.value = []; chartRows.value = []
  Object.keys(files).forEach(k => delete files[k]);unassignedFiles.value=[]; search.value = ''; hotspot.value = false; callFilter.value = {}; trail.value=[]; editingName.value=false; callSearch.value='';callStatus.value='';focus.value=''; applyView(view || {detailed:detailed.value}); if(view) {
    applyView(view)
    if(callFilter.value.related_type) trail.value.push({label:'类型 '+callFilter.value.related_type,view:{...snapshot(),tab:'overview',scope:{},callFilter:{},focus:''}})
    if(callFilter.value.group_id) trail.value.push({label:'整体分组',view:{...snapshot(),tab:'whole',callFilter:{...callFilter.value,group_id:undefined},recordPage:0,focus:''}})
  }
  Object.assign(options, { gpu_pid: '', npu_pid: '', gpu_step: '', npu_step: '', all_steps: false, include_communication: false, diagnostic_coverage_threshold: 90 }, next.options || {})
  options.gpu_step=normalizeStep(options.gpu_step);options.npu_step=normalizeStep(options.npu_step)
  for (const side of ['gpu', 'npu']) { const list = candidates(side); if (!options[side + '_pid'] && list.length === 1) options[side + '_pid'] = list[0].pid }
  if (next.status === 'succeeded') {
    await loadResults()
    if (focus.value) {
      const params=focus.value.startsWith('npu:')?{kind:'mappings',npu_index:focus.value.slice(4),limit:1}:{kind:'call_details',call_id:focus.value,limit:1}
      const found=await api(resultUrl(params)); if(found.items.length) await showRecord(found.items[0])
    }
  }
  writeLocation()
}
function candidates(side) { return job.value?.inspection?.profiles?.[side]?.cpu_candidates || [] }
function steps(side) { return candidates(side).find(c => c.pid === options[side + '_pid'])?.steps || [] }
async function createJob() { await action(async () => { const j = await api('/jobs', 'POST', { name: name.value.trim() || 'GPU / NPU comparison' }); name.value = ''; await refresh(); await select(j.id) }) }
function sendFile(id, role, file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); uploadRequest=xhr; xhr.onabort=()=>reject(new Error('上传已取消；已完成的文件保留，可继续上传')); xhr.open('PUT', `/api/jobs/${id}/inputs/${role}`); xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name)); xhr.setRequestHeader('Content-Type', 'application/octet-stream')
    xhr.upload.onprogress = e => { uploadPercent.value = e.lengthComputable ? Math.round(e.loaded / e.total * 100) : 0 }
    xhr.onload = () => { if (xhr.status >= 200 && xhr.status < 300) resolve(); else { let message = '上传失败'; try { message = JSON.parse(xhr.responseText).detail } catch {} reject(new Error(message)) } }
    xhr.onerror = () => reject(new Error('上传连接中断，可以重新上传该文件')); xhr.send(file)
  })
}
async function uploadAll() { await action(async () => {
  const id = job.value.id
  job.value=await api('/jobs/'+id)
  try { for (const role of roles) { const file = files[role.key]; if (!file) { if (!job.value.inputs?.[role.key]) throw new Error('请提供 ' + role.title); continue } if (file.size > config.value.max_upload_bytes) throw new Error(role.title + ' 超过上传上限'); uploading.value = role.title; uploadPercent.value = 0; await sendFile(id, role.key, file); delete files[role.key]; job.value=await api('/jobs/'+id) } await api(`/jobs/${id}/inspect`, 'POST'); await select(id); await refresh() } finally { uploading.value = ''; uploadRequest=null }
}) }
async function run() { await action(async () => { tab.value='overview'; const payload = { ...options }; if (payload.all_steps) { payload.gpu_step = ''; payload.npu_step = '' } job.value = await api(`/jobs/${job.value.id}/run`, 'POST', payload); await refresh() }) }
async function cancel() { await action(async () => { job.value = await api(`/jobs/${job.value.id}/cancel`, 'POST'); await refresh() }) }
async function removeJob() { if (!confirm('永久删除此任务的输入、所有运行结果和日志？此操作不可恢复。')) return; await action(async () => { await api(`/jobs/${job.value.id}?confirm=true`, 'DELETE'); job.value = null; localStorage.removeItem('profiling-job'); await refresh() }) }
function resultUrl(params) { return `/jobs/${job.value.id}/results?` + new URLSearchParams(Object.fromEntries(Object.entries(params).filter(([,v]) => v !== null && v !== undefined))) }
async function loadResults() {
  if (job.value?.status !== 'succeeded') return
  const id=job.value.id, sequence=++resultsSequence
  loadingResults.value=true;rows.value=[]
  try {
    const params={search:search.value,sort:sort.value,limit:30,offset:page.value*30,...(tab.value==='coverage'?(scope.value||{}):{}),...(hotspot.value?{coverage_max:90,npu_ratio_min:1}:{})}
    const chartParams={...params,sort:'npu_time',limit:200,offset:0}, key=id+job.value.attempt+JSON.stringify(chartParams)
    const [data,chart]=await Promise.all([api(resultUrl(params)),chartCache.has(key)?Promise.resolve(chartCache.get(key)):api(resultUrl(chartParams))])
    if(job.value?.id!==id||sequence!==resultsSequence) return
    if(chartCache.size>20) chartCache.clear();chartCache.set(key,chart)
    rows.value=data.items;total.value=data.total;chartRows.value=chart.items.filter(x=>x.row_kind==='npu_type')
    if(tab.value!=='overview') await loadRecords()
  } finally {if(sequence===resultsSequence) loadingResults.value=false}
}
async function loadRecords() {
  if(job.value?.status!=='succeeded'||tab.value==='overview') return
  const id=job.value.id,sequence=++recordsSequence
  loadingRecords.value=true;records.value=[];recordTotal.value=0
  try {
    const result=await api(resultUrl({...recordParams(),offset:recordPage.value*20,limit:20}))
    if(job.value?.id!==id||sequence!==recordsSequence)return
    records.value=result.items;recordTotal.value=result.total
  } finally {if(sequence===recordsSequence)loadingRecords.value=false}
}
async function openType(row) { remember('类型对比');scope.value={op_type:row.op_type,device_id:row.device_id,core_type:row.core_type};tab.value='coverage';page.value=0;recordPage.value=0;recordKind.value='reasons';writeLocation(true);await action(loadResults) }
async function openWhole(row) { remember('类型 '+row.op_type);callFilter.value={related_type:row.op_type,device_id:row.device_id,core_type:row.core_type};tab.value='whole';callSearch.value='';callStatus.value='';recordPage.value=0;writeLocation(true);await action(loadRecords) }
async function openCallGroup(row) { remember('整体 '+row['Whole OP']);callFilter.value={...callFilter.value,group_id:row['Group ID']};tab.value='calls';callSearch.value='';callStatus.value='';recordPage.value=0;writeLocation(true);await action(loadRecords) }
async function showRecord(record) {
  detail.value=record;evidence.value=null
  const index=record['NPU Kernel Index']??record.npu_kernel_index, id=job.value.id
  focus.value=record['Call ID']||(index!==undefined?'npu:'+index:'');writeLocation()
  if(record['Call ID']) await action(async()=>{
    const callId=record['Call ID'], result=await api('/jobs/'+id+'/call-evidence?'+new URLSearchParams({call_id:callId,limit:200}))
    if(job.value?.id===id&&focus.value===callId)evidence.value=result
  })
  else if(index!==undefined) await action(async()=>{
    const result=await api(resultUrl({kind:'diagnoses',npu_index:index,limit:5}))
    if(job.value?.id===id&&focus.value==='npu:'+index)evidence.value=result
  })
}
function openReport() { window.location.href=download('coverage_diagnostics.md') }
function download(name) { return `/api/jobs/${job.value.id}/artifacts/${encodeURIComponent(name)}` }
async function switchTab(value) { remember(tabs.find(t=>t[0]===tab.value)?.[1]||'返回'); tab.value=value; recordPage.value=0;callFilter.value={};scope.value=null;focus.value='';writeLocation(true);await action(loadResults) }
async function filter() { page.value=0;recordPage.value=0;await action(loadResults) }
function clickedChart(p) { const row = chartRows.value.find(x => `${x.op_type} · ${x.device_id}/${x.core_type}` === p.name); if (row) openType(row) }
const tooltip = { trigger: 'axis', renderMode: 'richText', confine: true }
const bars = computed(() => { const data = chartRows.value.slice(0, 12).reverse(); return { tooltip, legend: { data: ['NPU 总时间', '已分配 GPU 时间'] }, grid: { left: 160, right: 30, top: 40, bottom: 35 }, xAxis: { type: 'value', name: 'μs' }, yAxis: { type: 'category', data: data.map(r => `${r.op_type} · ${r.device_id}/${r.core_type}`), axisLabel: { width: 145, overflow: 'truncate' } }, series: [{ name: 'NPU 总时间', type: 'bar', data: data.map(r => r.npu_total_time_us), itemStyle: { color: '#167d78' } }, { name: '已分配 GPU 时间', type: 'bar', data: data.map(r => r.gpu_total_time_us), itemStyle: { color: '#7489ba' } }] } })
const scatter = computed(() => ({ tooltip: { renderMode: 'richText', confine: true }, grid: { left: 60, right: 40, top: 20, bottom: 50 }, xAxis: { name: 'NPU 时间占比 %', nameLocation: 'middle', nameGap: 30, type: 'value' }, yAxis: { name: 'Comparable 时间覆盖率 %', max: 100, min: 0 }, series: [{ type: 'scatter', itemStyle: { color: '#167d78', opacity: .7 }, data: chartRows.value.map(r => ({ name: `${r.op_type} · ${r.device_id}/${r.core_type}`, value: [r.npu_ratio_pct, r.coverage.comparable.time_pct], symbolSize: Math.min(32, 8 + Math.sqrt(Math.max(0, r.npu_total_time_us - r.npu_comparable_time_us))) })) }] }))
watch(recordKind, () => { if(!restoring && tab.value==='coverage') {recordPage.value=0;action(loadRecords)} })
watch(snapshot,()=>writeLocation(),{deep:true})
watch(job,()=>{if(!job.value){history.replaceState({},'',location.pathname);trail.value=[]}})
onMounted(async () => {
  try { await refresh(); logged.value=true; await restoreLocation() } catch(e) { if(logged.value)error.value=e.message }
  window.addEventListener('popstate',onPopState)
  timer = setInterval(async () => { if (!logged.value || loading || busy.value || document.hidden || (!active.value && Date.now()-lastRefresh<30000)) return; loading = true; try { await refresh(); if (job.value) { const before = job.value.status; const id = job.value.id; const next = await api('/jobs/' + id); if (job.value?.id === id) { job.value = next; elapsed.value = next.started ? Math.round((Date.now() / 1000 - next.started)) : 0; if (before !== next.status) { if (next.status === 'succeeded') await loadResults(); if (next.status === 'needs_selection') for (const side of ['gpu', 'npu']) if (candidates(side).length === 1) options[side + '_pid'] = candidates(side)[0].pid } } } } catch(e) { error.value = e.message } finally { loading = false } }, 2500)
})
onBeforeUnmount(() => { clearInterval(timer);window.removeEventListener('popstate',onPopState) })
</script>

<template>
  <div v-if="!logged" class="login-page"><section class="login-card"><div class="brand-mark">P<span>↗</span></div><p class="eyebrow">LOCAL ANALYSIS WORKSPACE</p><h1>Profiling Workbench</h1><p class="muted">把 GPU 与 NPU 的性能差异，追溯到每一次调用。</p><form @submit.prevent="login"><label>访问令牌<input v-model="token" type="password" autocomplete="current-password" placeholder="粘贴服务器访问令牌" required></label><button class="primary" :disabled="busy">进入工作台 →</button></form><p class="muted small">令牌位于服务器数据目录的 access-token 文件，或由管理员提供。此版本是单机共享工作区。</p><p v-if="error" class="error">{{ error }}</p></section></div>
  <div v-else class="app-shell">
    <aside><div class="brand"><div class="brand-mark">P<span>↗</span></div><div>Profiling<span>WORKBENCH</span></div></div><div class="workspace-label">单机 · 共享工作区</div><form class="new-job" @submit.prevent="createJob"><input v-model="name" placeholder="比较任务名称" maxlength="120" aria-label="新任务名称"><button class="primary" :disabled="busy">＋ 新建比较</button></form><div class="section-label">任务 <span>{{ jobTotal }}</span></div><input v-model="jobSearch" placeholder="搜索任务名" aria-label="搜索任务名" @keyup.enter="findJobs"><div class="task-filters"><select v-model="jobStatus" @change="findJobs" aria-label="任务状态"><option value="">全部状态</option><option v-for="s in ['succeeded','running','queued','failed','needs_selection','interrupted','cancelled','uploading']" :key="s" :value="s">{{ labels[s] }}</option></select><label class="check"><input type="checkbox" v-model="jobArchived" @change="findJobs">归档</label><button @click="findJobs">搜索</button></div><nav><button v-for="j in jobs" :key="j.id" :class="['job-link', { selected: job?.id === j.id }]" @click="action(() => select(j.id))" :disabled="busy"><span class="job-name">{{ j.name }}</span><span class="job-status"><i :class="j.status"></i>{{ labels[j.status] }} · {{ new Date(j.created * 1000).toLocaleDateString() }}</span></button></nav><div class="pagination"><button :disabled="jobPage===0 || busy" @click="jobPage--;action(refresh)">上一页</button><span>{{ jobPage+1 }}</span><button :disabled="(jobPage+1)*30>=jobTotal || busy" @click="jobPage++;action(refresh)">下一页</button></div><footer><span :class="['worker-dot', { online: config.worker_online }]"></span>{{ config.worker_online ? 'Worker 在线 · 并发 1' : 'Worker 离线，请检查服务' }}<button class="text-button" v-if="config.auth_mode !== 'local'" @click="logout">退出</button><span v-else>本机免令牌</span></footer></aside>
    <main><header><div><p class="eyebrow">GPU / ASCEND NPU</p><form v-if="editingName" class="rename-form" @submit.prevent="renameJob"><input v-model="editedName" aria-label="任务名称" maxlength="120" required><button class="primary" :disabled="busy || !editedName.trim()">保存</button><button type="button" @click="editingName=false">取消</button></form><h1 v-else>{{ job?.name || '性能差异，从证据开始' }} <button v-if="job" class="text-button" @click="editedName=job.name;editingName=true">修改名称</button></h1><p class="muted">{{ job ? '任务 ' + job.id.slice(0, 8) + ' · ' + (labels[job.status] || job.status) : '上传两端采集数据，生成可追溯的对齐结果与覆盖率分析。' }}</p></div><span class="privacy">● 本地处理 · 无外部 AI 调用</span></header>
      <div class="service-toolbar"><button @click="showService">服务状态</button><span>{{ config.auth_mode==='local'?'本机免令牌访问':'令牌认证 · 共享工作区' }}</span></div><section v-if="service" class="panel"><h2>服务状态</h2><p>Worker {{ service.worker_online?'在线':'离线' }} · 排队 {{ service.tasks.queued || 0 }} · 运行 {{ service.tasks.running || 0 }}</p><p>存储 {{ filesize(service.storage_used_bytes) }} / {{ filesize(service.storage_bytes) }} · 内存软限制 {{ service.memory_mb }} MiB · 超时 {{ service.timeout_seconds }} 秒</p><p class="muted">状态为打开时快照。内存上限不是当前内存用量。</p></section><div v-if="error" class="error banner" role="alert">{{ error }}<button @click="error = ''" aria-label="关闭错误">×</button></div>
      <section v-if="!job" class="empty panel"><div class="empty-symbol">↗</div><h2>创建第一组性能比较</h2><p>准备 GPU trace、NPU trace 和 kernel_details.csv。<br>在左侧新建任务，选择比较范围后开始分析。</p><div class="steps"><span>01 上传数据</span><span>02 确认范围</span><span>03 分析与查证</span></div></section>
      <template v-else>
        <div class="toolbar"><span class="muted">输入和输出仅保存在此服务器的共享工作区。</span><button v-if="!active && job.status !== 'uploading'" :disabled="busy" @click="cloneJob">复用输入，新建分析</button><button v-if="!active" :disabled="busy || !!job.uploading" @click="archiveJob">{{ job.archived?'取消归档':'归档任务' }}</button><button v-if="!active" :disabled="busy || !!job.uploading" @click="removeJob">删除任务…</button></div>
        <section v-if="job.status === 'uploading'" class="panel"><div class="panel-heading"><h2>01 / 上传采集数据</h2><span class="muted">单文件上限 {{ filesize(config.max_upload_bytes) }}</span></div><div class="drop-zone" @dragover.prevent @drop.prevent="acceptFiles"><strong>拖拽多个文件到此处</strong><p class="muted">按文件名建议角色，请检查下方分配。无法识别的文件可手动选择角色。</p><input type="file" multiple accept=".json,.csv" aria-label="批量选择输入文件" :disabled="busy" @change="acceptFiles"></div><div v-for="(f,i) in unassignedFiles" :key="i" class="unassigned">{{ f.name }} <select aria-label="分配文件角色" @change="assignFile(i,$event.target.value)"><option value="">选择文件角色</option><option v-for="r in roles" :key="r.key" :value="r.key">{{ r.title }}</option></select><button @click="unassignedFiles.splice(i,1)">移除</button></div><div class="upload-grid"><label v-for="role in roles" :key="role.key" class="upload-box"><span class="file-icon">{{ role.accept === '.csv' ? 'CSV' : 'JSON' }}</span><strong>{{ role.title }}</strong><span class="muted">{{ role.hint }}</span><input type="file" :accept="role.accept" :disabled="busy" @change="files[role.key] = $event.target.files[0]"><span class="small">{{ files[role.key]?.name || job.inputs?.[role.key]?.name || '请选择文件' }}</span><span v-if="job.inputs?.[role.key]" class="success">已保存 · {{ filesize(job.inputs[role.key].size) }}</span></label></div><p class="muted">文件将流式保存到服务器；不会在浏览器解析完整 trace。输入开始预检查后冻结。</p><div v-if="uploading" class="upload-progress">{{ uploading }} · {{ uploadPercent }}%<progress :value="uploadPercent" max="100"></progress><button @click="cancelUpload">取消当前上传</button></div><button class="primary" :disabled="busy" @click="uploadAll">{{ busy ? '正在上传…' : '上传并预检查 →' }}</button></section>
        <section v-if="active" class="panel running"><div class="spinner"></div><div><h2>{{ labels[job.stage] || job.stage }}</h2><p class="muted">{{ job.status === 'queued' ? '任务已持久化排队，可安全刷新页面。' : `已运行约 ${elapsed} 秒。阶段进度不是预估百分比。` }}</p></div><button :disabled="busy || !!job.cancel" @click="cancel">{{ job.cancel ? '正在取消…' : '取消任务' }}</button></section>
        <div v-if="job.error" class="error banner"><strong>{{ job.error }}</strong><p>{{ errorAdvice }}</p></div>
        <section v-if="job.inspection && job.status !== 'succeeded' && !active" class="panel"><div class="panel-heading"><h2>02 / 确认比较范围</h2><span class="tag">{{ job.inspection.status }}</span></div><ul class="warnings"><li v-for="(w, i) in [...job.inspection.errors, ...job.inspection.warnings, ...job.inspection.selections_required]" :key="i">{{ w.side }} · {{ w.message }}</li></ul><div class="scope-grid"><div v-for="side in ['gpu', 'npu']" :key="side"><h3>{{ side.toUpperCase() }}</h3><label>CPU PID<select v-model="options[side + '_pid']" @change="options[side + '_step'] = ''"><option value="">选择进程</option><option v-for="c in candidates(side)" :key="c.pid" :value="c.pid">{{ c.pid }} · {{ c.op_count }} CPU ops</option></select></label><label>Step<select v-model="options[side + '_step']" :disabled="options.all_steps"><option value="">选择稳定 step</option><option v-for="(s, i) in steps(side)" :key="i" :value="normalizeStep(s.name)">{{ s.name }} · {{ fmt(s.duration_us) }} μs</option></select></label></div></div><div class="options"><label class="check"><input type="checkbox" v-model="options.all_steps">明确比较完整采集范围（包括 warmup / 未标记事件）</label><label class="check"><input type="checkbox" v-model="options.include_communication">包含通信 kernel</label><label>诊断阈值 %<input type="number" v-model.number="options.diagnostic_coverage_threshold" min="0" max="100"></label></div><p class="notice">相同 step 编号不证明输入和工作量等价。请确认 batch、shape、精度及设备范围一致；PID 不是设备过滤器。</p><button class="primary" :disabled="busy || !selectable || job.inspection.status === 'invalid'" @click="run">开始分析 →</button></section>
        <section v-if="['failed', 'cancelled', 'interrupted'].includes(job.status) && !job.inspection" class="panel"><p>可重试预检查。若原始输入有误，请新建任务上传修正后的文件。</p><button @click="action(async () => { job = await api(`/jobs/${job.id}/inspect`, 'POST') })">重试预检查</button></section>
        <template v-if="job.status === 'succeeded' && summary">
          <p class="notice">全任务比较范围（不随下方筛选改变）：GPU PID {{ job.options.gpu_pid || '自动选择' }} / {{ job.options.all_steps ? '完整采集' : job.options.gpu_step }}；NPU PID {{ job.options.npu_pid || '自动选择' }} / {{ job.options.all_steps ? '完整采集' : job.options.npu_step }}。耗时图沿用类型表的当前筛选条件。</p>
          <div class="result-meta"><span class="tag success">✓ {{ summary.saved_validation?.checks_run }} 项校验通过</span><span class="muted">引擎 {{ summary.producer?.version }} · {{ summary.status === 'complete_with_gaps' ? '已完成，仍有覆盖缺口' : '已完成' }}</span><a class="button primary" :href="download('npu_gpu_op_statistic.xlsx')">↓ 下载 Excel</a><details class="downloads"><summary>更多输出</summary><a v-for="file in summary.reports" :key="file" :href="download(file)">{{ file }}</a></details></div>
          <div class="metrics"><article><span>NPU 总时间</span><strong>{{ fmt(overall?.npu_total_time_us) }}<small> μs</small></strong><p>kernel_details-backed</p></article><article><span>GPU 总时间</span><strong>{{ fmt(overall?.gpu_total_time_us) }}<small> μs</small></strong><p>计算 kernel + 内存活动，去重</p></article><article><span>Comparable Ratio · GPU/NPU</span><strong :class="ratioClass(overall?.comparable_ratio_gpu_npu)">{{ fmt(overall?.comparable_ratio_gpu_npu, 3) }}<small> ×</small></strong><p>对应可比子集，不包含所有工作</p></article><article><span>Total Time Ratio · GPU/NPU</span><strong :class="ratioClass(overall?.total_time_ratio_gpu_npu)">{{ fmt(overall?.total_time_ratio_gpu_npu, 3) }}<small> ×</small></strong><p>总体范围比较，非等价工作量证明</p></article></div>
          <CoreBreakdown :data="summary.core_breakdown" />
          <details class="notice"><summary>统计口径与 {{ summary.warnings.length }} 条警告</summary><p>耗时是事件时长之和，不是墙钟时间。NPU trace-only 任务不计入 TOTAL；GPU 未映射计算与内存活动计入。整体算子视图与类型视图重叠，不能累加。</p><p v-for="(w, i) in summary.warnings" :key="i">{{ w.message }} <span v-if="w.count">数量 {{ w.count }}，{{ fmt(w.time_us) }} μs</span></p></details>
          <div class="tabs" role="tablist"><button v-for="[key, label] in tabs" :key="key" :class="{ active: tab === key }" @click="switchTab(key)" role="tab" :aria-selected="tab === key">{{ label }}</button></div>
          <section class="panel result-panel">
            <nav v-if="trail.length" class="breadcrumbs" aria-label="分析路径"><button v-for="(item,i) in trail" :key="i" @click="backTo(i)">{{ item.label }} ←</button><span>{{ tabs.find(t=>t[0]===tab)?.[1] }}</span></nav>
            <div class="toolbar"><label class="check"><input type="checkbox" v-model="detailed">详细模式</label><select v-model="savedViewIndex" @change="useView" aria-label="已保存视图"><option value="">选择已保存视图</option><option v-for="(v,i) in savedViews" :key="i" :value="String(i)">{{ v.name }}</option></select><button @click="savingView=!savingView">保存视图</button><form v-if="savingView" class="toolbar" @submit.prevent="saveView"><input v-model="viewName" aria-label="视图名称" placeholder="输入视图名称" maxlength="60" required><button type="submit">确认保存视图</button><button type="button" @click="savingView=false">取消</button></form><button v-if="savedViewIndex!==''" @click="deleteView">删除视图</button><button :disabled="busy" @click="exportView">导出当前筛选 CSV</button></div>
            <p class="muted small">视图设置只影响展示。分析范围和诊断生成阈值的变更需“复用输入，新建分析”。</p>
            <p v-if="loadingResults || loadingRecords" role="status">正在加载结果…</p>
            <template v-if="tab === 'overview' || tab === 'coverage'"><div class="toolbar"><input v-model="search" placeholder="搜索 OP Type" @keyup.enter="filter" aria-label="搜索类型"><select v-model="sort" @change="filter" aria-label="排序"><option value="npu_time">NPU 耗时降序</option><option value="coverage">可比时间覆盖率升序</option><option value="ratio">可比性能比降序</option><option value="name">类型名称</option></select><label class="check"><input type="checkbox" v-model="hotspot" @change="filter">NPU 占比 &gt; 1% 且可比时间覆盖率 &lt; 90%</label><button @click="filter">筛选</button></div><Chart v-if="tab === 'coverage'" :option="scatter" @select="clickedChart"/><p v-if="tab === 'coverage'" class="muted small">气泡大小表示不可比较耗时，不代表可节省时间。图表最多展示筛选后前 200 个类型。</p><p class="filter-context">当前类型范围：{{ scope ? scope.op_type+' / '+scope.device_id+' / '+scope.core_type : '全部类型' }} · 搜索 {{ search || '不限' }} · {{ hotspot?'高耗时低覆盖率':'不限覆盖率' }} <button @click="clearFilters">清除筛选</button></p><TypeTable v-if="!loadingResults" :rows="rows" :detailed="detailed" @type="openType" @whole="openWhole" @report="openReport" @manual="manualType=$event"/><div class="pagination"><span>共 {{ total }} 行 · 第 {{ page + 1 }} 页</span><button :disabled="page === 0" @click="page--; action(loadResults)">上一页</button><button :disabled="(page + 1) * 30 >= total" @click="page++; action(loadResults)">下一页</button></div></template>
            <template v-if="tab === 'distribution'"><h2>类型耗时 Top 12</h2><p class="muted">NPU 全类型时间与已分配 GPU 时间并列；覆盖率低时不是完整同类性能比较。点击图形查看类型证据。</p><Chart :option="bars" @select="clickedChart"/><div class="accounting"><div v-for="kind in ['kernel', 'memory']" :key="kind"><h3>GPU {{ kind === 'kernel' ? '计算 kernel' : '内存活动' }}</h3><p>已映射 {{ summary.gpu_accounting[kind].mapped_count }} 个 / {{ fmt(summary.gpu_accounting[kind].mapped_time_us) }} μs</p><p>未映射 {{ summary.gpu_accounting[kind].unmapped_count }} 个 / {{ fmt(summary.gpu_accounting[kind].unmapped_time_us) }} μs</p></div></div><h3>GPU 未映射事件</h3></template>
            <template v-if="tab === 'coverage'"><div class="panel-heading evidence-heading"><h2>{{ scope ? `${scope.op_type} · ${scope.device_id}/${scope.core_type}` : '所有低覆盖率类型' }} / 证据</h2><button v-if="scope" @click="clearFilters()">清除类型选择</button></div><div class="toolbar"><select v-model="recordKind" aria-label="证据类型"><option value="reasons">原因汇总与排查建议</option><option value="mappings">逐 kernel 映射</option><option value="diagnoses">逐 kernel 诊断</option></select><span class="muted">Accounted 不等于 Comparable；原因是候选解释，不自动证明后端缺陷。</span></div></template>
            <template v-if="tab === 'whole' || tab === 'calls'">
              <h2>{{ tab === 'whole' ? '完整调用性能比较' : '完整调用明细' }}</h2>
              <p class="notice">整体视图与类型表以及重叠父子调用不可相加，不提供跨行 TOTAL。GPU 时间包括该调用的计算及内存活动。Div/Sub 专项路径只在排查报告中。</p>
              <p v-if="!supportsWhole" class="notice">这是旧版报告，未生成整体调用视图。请复用输入并重新分析；旧结果保持不变。</p>
              <div class="toolbar"><input v-model="callSearch" placeholder="搜索逻辑调用名称" aria-label="搜索调用名称" @keyup.enter="recordPage=0;action(loadRecords)"><select v-model="callStatus" @change="recordPage=0;action(loadRecords)" aria-label="比较状态"><option value="">全部比较状态</option><option value="comparable">包含可比调用</option><option value="pending">包含待核实调用</option></select><select v-model="callSort" @change="recordPage=0;action(loadRecords)" aria-label="调用排序"><option value="npu_time">NPU 耗时降序</option><option value="ratio">可比比值降序</option><option value="name">名称</option></select><button @click="recordPage=0;action(loadRecords)">筛选调用</button></div>
              <div v-if="Object.keys(callFilter).length" class="toolbar"><span>当前关联：{{ callFilter.related_type || callFilter.group_id }}</span><button @click="callFilter = {}; recordPage = 0; action(loadRecords)">清除关联筛选</button></div>
              <WholeCallTable v-if="!loadingRecords && supportsWhole" :rows="records" :details="tab === 'calls'" @group="openCallGroup" @detail="showRecord"/>
            </template>
            <template v-if="tab !== 'overview'"><div v-if="!loadingRecords && tab !== 'whole' && tab !== 'calls'" class="record-list"><article v-for="(r, i) in records" :key="i" class="record"><template v-if="tab === 'coverage' && recordKind === 'reasons'"><div class="record-heading"><strong>{{ r.op_type }} · {{ r.reason }}</strong><span>{{ r.kernel_count }} 个 · {{ fmt(r.npu_time_us) }} μs</span></div><p>{{ r.hypothesis }}</p><p class="small muted">CPU {{ r.cpu_op || '未知' }} · API {{ r.npu_api || '未知' }} · NPU 样本 {{ r.sample_kernel_indices?.join(', ') }}</p><ol><li v-for="step in r.next_steps" :key="step">{{ step }}</li></ol></template><template v-else><button class="type-link" @click="showRecord(r)">{{ r['Fused OP'] || r['CPU Bundle'] || r['NPU Kernel Name'] || r['GPU Name'] || r.op_type || '事件' }} <span v-if="r['NPU Kernel Index'] !== undefined">#{{ r['NPU Kernel Index'] }}</span></button><p v-if="r.cpu_alignment_status">{{ r.cpu_alignment_status }} · {{ r.kernel_comparison_status }}</p><p class="muted small">{{ r.source || r['Mapping Mode'] || r.reason || r['Status'] || '查看结构化证据' }}</p><button @click="showRecord(r)">查看详情 →</button></template></article><p v-if="!records.length" class="muted">没有对应记录。无诊断可能是类型未命中阈值，不代表已证明等价。</p></div><div class="pagination"><span>共 {{ recordTotal }} 条 · 第 {{ recordPage + 1 }} 页</span><button :disabled="recordPage === 0" @click="recordPage--; action(loadRecords)">上一页</button><button :disabled="(recordPage + 1) * 20 >= recordTotal" @click="recordPage++; action(loadRecords)">下一页</button></div></template>
          </section>
        </template>
      </template>
      <footer class="main-footer">PROFILING WORKBENCH <span>证据优先 · 独立计时 · 可追溯结果</span></footer>
    </main>
    <EvidenceDrawer v-if="detail" :record="detail" :evidence="evidence" @close="closeDetail"/>
    <ManualLinkDialog v-if="manualType" :key="job.id" :job="job" :type="manualType" :api="api" @close="manualType=null" @submitted="manualSubmitted" @upgrade="manualUpgrade"/>
  </div>
</template>
