<script setup>
import { computed, onMounted, onBeforeUnmount, ref, reactive, watch, nextTick } from 'vue'
import NpuPairPanel from './NpuPairPanel.vue'
import Chart from './Chart.vue'
import WholeCallTable from './WholeCallTable.vue'
import TypeTable from './TypeTable.vue'
import EvidenceDrawer from './EvidenceDrawer.vue'
import ManualLinkDialog from './ManualLinkDialog.vue'
import AiAssistant from './AiAssistant.vue'
import CoreBreakdown from './CoreBreakdown.vue'
import ParameterSummary from './ParameterSummary.vue'
import ReportDownloads from './ReportDownloads.vue'
import {exportCsv} from './tableExport.js'
import OperatorDetails from './OperatorDetails.vue'
import { cleanView, readLocation, viewQuery, readSavedViews, normalizeStep } from './viewState.js'
const logged = ref(false), token = ref(''), error = ref(''), busy = ref(false), config = ref({})
const jobs = ref([]), job = ref(null), tab = ref('overview'), name = ref(''), files = reactive({}), uploading = ref(''), uploadPercent = ref(0)
const newMode = ref('gpu_npu'), isPair = computed(() => job.value?.mode === 'npu_pair')
const createDialog = ref(null)
// The two flows take different inputs and cannot be switched after creation, so the
// choice lives in the creation dialog rather than a bare <select>.
const modes = [
  { id: 'gpu_npu', from: 'GPU', to: 'NPU', title: 'GPU ↔ NPU 实测对比', tag: '双侧实测',
    summary: '对齐 kernel，量化覆盖率与实测性能比。',
    inputs: ['GPU Trace', 'NPU Trace', 'kernel_details.csv'],
    hint: '创建后上传 3 份采集文件，再确认 PID 与 Step 范围。',
    placeholder: '例如 BERT-base · A100 与 910B · Step 15' },
  { id: 'npu_pair', from: 'NPU', to: 'NPU', title: 'NPU ↔ NPU 前后对比', tag: '仅需明细',
    summary: '比较前后性能比、CV 结构与流水初筛。',
    inputs: ['基线 kernel_details.csv', '对比 kernel_details.csv'],
    hint: '创建后上传 2 份明细，再选择各自的 Device 与 Step。',
    placeholder: '例如 升级 CANN 前后 · Step 15' },
]
const currentMode = computed(() => modes.find(m => m.id === newMode.value) || modes[0])
function startNew(mode) { newMode.value = mode; openCreate() }
function openCreate() { createDialog.value?.showModal() }
function closeCreate() { createDialog.value?.close() }
const callFilter = ref({})
const callSearch = ref(''), callSort = ref('npu_time'), callStatus = ref(''), detailed = ref(false), focus = ref('')
const jobSearch = ref(''), jobStatus = ref(''), jobArchived = ref(false), jobPage = ref(0), jobTotal = ref(0)
const editingName = ref(false), editedName = ref(''), service = ref(null), loadingResults = ref(false), loadingRecords = ref(false)
const savedViews = ref(readSavedViews(localStorage.getItem('profiling-views-v1'))), savedViewIndex = ref(''), viewName = ref(''), savingView = ref(false)
const trail = ref([]), chartCache = new Map()
const manualType = ref(null), manualSeed=ref(null), aiScope=ref(null)
function openAI(row={}) {aiScope.value=row;tab.value='ai';writeLocation(true)}
function openManual(row) {manualSeed.value=null;manualType.value=row}
function aiAssociate(value) {manualType.value=value.type;manualSeed.value=value.seed;}
const operatorState = ref(cleanView().operator)
watch(()=>job.value?.id,()=>{aiScope.value=null;manualType.value=null;manualSeed.value=null;shareLink.value='';shareCopied.value=false})
function manualSubmitted(next) {job.value=next;manualType.value=null;manualSeed.value=null;chartCache.clear();action(refresh)}
function manualUpgrade() {manualType.value=null;cloneJob()}
let restoring = false, lastRefresh = 0
const snapshot = () => cleanView({tab:tab.value,search:search.value,sort:sort.value,hotspot:hotspot.value,page:page.value,recordPage:recordPage.value,recordKind:recordKind.value,scope:scope.value,callFilter:callFilter.value,callSearch:callSearch.value,callSort:callSort.value,callStatus:callStatus.value,detailed:detailed.value,focus:focus.value,operator:operatorState.value,aiScope:aiScope.value})
function applyView(v) {
  const x=cleanView(v); tab.value=x.tab; search.value=x.search; sort.value=x.sort; hotspot.value=x.hotspot; page.value=x.page; recordPage.value=x.recordPage; recordKind.value=x.recordKind
  operatorState.value=x.operator;aiScope.value=x.aiScope
  scope.value=Object.keys(x.scope).length ? x.scope : null; callFilter.value=x.callFilter; callSearch.value=x.callSearch; callSort.value=x.callSort; callStatus.value=x.callStatus; detailed.value=x.detailed; focus.value=x.focus
}
function writeLocation(push=false) { if (!restoring && job.value) history[push?'pushState':'replaceState']({},'', '?' + viewQuery(job.value.id,snapshot())) }
function remember(label) { trail.value.push({label,view:snapshot()}); focus.value='' }
async function backTo(index) { const item=trail.value[index]; trail.value=trail.value.slice(0,index); applyView(item.view); detail.value=null; writeLocation(true); await action(loadResults) }
function saveView() { const label=viewName.value; if (!label?.trim()) return; const view={...snapshot(), page:0, recordPage:0, scope:{}, callFilter:{}, focus:'', operator:{...operatorState.value,page:0,unitPage:0,group:'',unit:'',attempt:''}}; savedViews.value=[...savedViews.value.filter(v=>v.name!==label.trim()),{name:label.trim().slice(0,60),view}].slice(-20); localStorage.setItem('profiling-views-v1',JSON.stringify(savedViews.value)); savingView.value=false; viewName.value=''; savedViewIndex.value=String(savedViews.value.length-1) }
async function useView() { if(savedViewIndex.value==='') return; const v=savedViews.value[Number(savedViewIndex.value)]; if (!v) return; remember('上一个视图'); applyView(v.view); writeLocation(true); await action(loadResults) }
function deleteView() { if(savedViewIndex.value==='') return; savedViews.value.splice(Number(savedViewIndex.value),1); localStorage.setItem('profiling-views-v1',JSON.stringify(savedViews.value)); savedViewIndex.value='' }
async function renameJob() { await action(async()=>{ job.value=await api('/jobs/'+job.value.id,'PATCH',{name:editedName.value.trim()}); editingName.value=false; await refresh() }) }
async function archiveJob() { await action(async()=>{ job.value=await api('/jobs/'+job.value.id,'PATCH',{archived:!job.value.archived}); await refresh() }) }
async function cloneJob() { await action(async()=>{ const j=await api('/jobs/'+job.value.id+'/clone','POST'); jobArchived.value=false; jobSearch.value=''; jobPage.value=0; await refresh(); await select(j.id) }) }
// The view state is large; keep it on the server and share an 8-character link instead.
const shareLink = ref(''), shareCopied = ref(false), sharing = ref(false)
async function shareCurrentView() {
  if (!job.value || sharing.value) return
  sharing.value = true
  error.value = ''
  try {
    const result = await api('/short-links', 'POST', { job: job.value.id, view: JSON.stringify(snapshot()) })
    shareLink.value = location.origin + result.path
    await nextTick()
    await copyShare()
  } catch (e) { error.value = e.message } finally { sharing.value = false }
}
async function copyShare() {
  if (!shareLink.value) return
  let copied = false
  try { await navigator.clipboard.writeText(shareLink.value); copied = true } catch {}
  if (!copied) {
    // Plain-HTTP LAN access is not a secure context, so select the field for a manual copy.
    const field = document.querySelector('input[aria-label="分享短链接"]')
    if (field) { field.select(); try { copied = document.execCommand('copy') } catch {} }
  }
  shareCopied.value = copied
  setTimeout(() => { shareCopied.value = false }, 4000)
}
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
const aggregateSources = ref([])
watch(()=>[job.value?.id,job.value?.status], async ([id,status],_,cleanup)=>{
  let current=true;cleanup(()=>{current=false});aggregateSources.value=[]
  if (!isPair.value && status==='needs_selection') {
    try {const r=await api('/jobs/'+id+'/aggregate-sources');if(current)aggregateSources.value=r.items}
    catch(e){if(current)error.value=e.message}
  }
})
const supportsWhole = computed(()=>summary.value?.reports?.includes('whole_call_mapping.csv'))
async function exportView() { await action(async()=>{
  const id=job.value.id, state=snapshot(), params=['overview','coverage'].includes(tab.value)?{search:search.value,sort:sort.value,...(tab.value==='coverage'?(scope.value||{}):{}),...(hotspot.value?{coverage_max:90,npu_ratio_min:1}:{})}:recordParams()
  await exportCsv(api,'/jobs/'+id+'/results?'+new URLSearchParams(params),id.slice(0,8)+'-'+tab.value+'-filtered.csv',{scope:{job:id,view:state}})
}) }
const rows = ref([]), chartRows = ref([]), total = ref(0), page = ref(0), search = ref(''), sort = ref('npu_time'), hotspot = ref(false)
const detail = ref(null), evidence = ref(null), scope = ref(null), records = ref([]), recordTotal = ref(0), recordPage = ref(0), recordKind = ref('reasons')
const options = reactive({ gpu_pid: '', npu_pid: '', gpu_step: '', npu_step: '', all_steps: false, export_call_tables: false, aggregate_all_job: '', diagnostic_coverage_threshold: 90, include_communication: false })
const unassignedFiles=ref([])
let uploadRequest=null
function acceptFiles(event) {
  if(busy.value)return
  const incoming=[...(event.dataTransfer?.files || event.target.files || [])]
  for(const file of incoming) {
    const n=file.name.toLowerCase()
    const role=isPair.value ? null : n.endsWith('.csv')&&n.includes('kernel_details')?'npu_kernel_details':n.endsWith('.json')&&n.includes('gpu')?'gpu_trace':n.endsWith('.json')&&(n.includes('npu')||n.includes('trace_view'))?'npu_trace':null
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
const roles = computed(() => isPair.value ? [
  {key:'baseline_details', title:'基线 NPU · kernel_details.csv', hint:'作为分母的采集', accept:'.csv'},
  {key:'candidate_details', title:'对比 NPU · kernel_details.csv', hint:'要评估的采集', accept:'.csv'}
] : [{ key: 'gpu_trace', title: 'GPU trace', hint: 'PyTorch profiler · JSON', accept: '.json' }, { key: 'npu_trace', title: 'NPU trace', hint: 'Ascend trace_view · JSON', accept: '.json' }, { key: 'npu_kernel_details', title: 'NPU kernel details', hint: 'kernel_details · CSV', accept: '.csv' }])
const labels = { aggregate: '生成类型与调用方聚合表', call_review: '生成严格调用评审表', uploading: '等待上传', queued: '排队中', running: '分析中', needs_selection: '等待确认范围', succeeded: '分析完成', failed: '失败', cancelled: '已取消', interrupted: '已中断', inspection: '校验输入 / 预检查', cache_load: '检查可复用的解析结果', cache_write: '保存解析缓存', reuse_exports: '复用 Step 下载文件', exports: '导出 Step 数据', gpu_parse: '解析 GPU trace', npu_parse: '解析 NPU trace', detail_link: '关联 NPU 明细', matching: '匹配 CPU / kernel', reports: '生成报告', validation: '校验计时与映射', indexing: '构建查询索引', starting: '启动分析进程' }
const tabs = [['overview', '类型对比'], ['operator', '算子详情'], ['whole', '整体对比'], ['coverage', '覆盖率诊断'], ['calls', '调用明细'], ['distribution', '耗时分布'], ['ai', 'AI 分析']]
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
  Object.assign(options, { gpu_pid: '', npu_pid: '', gpu_step: '', npu_step: '', all_steps: false, export_call_tables: false, aggregate_all_job: '', include_communication: false, diagnostic_coverage_threshold: 90 }, isPair.value ? {} : (next.options || {}))
  options.gpu_step=normalizeStep(options.gpu_step);options.npu_step=normalizeStep(options.npu_step)
  for (const side of ['gpu', 'npu']) { const list = candidates(side); if (!options[side + '_pid'] && list.length === 1) options[side + '_pid'] = list[0].pid }
  if (next.status === 'succeeded' && !isPair.value) {
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
async function createJob() {
  const mode = newMode.value
  const label = name.value.trim() || (mode === 'npu_pair' ? 'NPU / NPU comparison' : 'GPU / NPU comparison')
  closeCreate()
  await action(async () => { const j = await api('/jobs', 'POST', { mode, name: label }); name.value = ''; await refresh(); await select(j.id) })
}
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
  try { for (const role of roles.value) { const file = files[role.key]; if (!file) { if (!job.value.inputs?.[role.key]) throw new Error('请提供 ' + role.title); continue } if (file.size > config.value.max_upload_bytes) throw new Error(role.title + ' 超过上传上限'); uploading.value = role.title; uploadPercent.value = 0; await sendFile(id, role.key, file); delete files[role.key]; job.value=await api('/jobs/'+id) } await api(`/jobs/${id}/inspect`, 'POST'); await select(id); await refresh() } finally { uploading.value = ''; uploadRequest=null }
}) }
async function run() { await action(async () => { tab.value='overview'; const payload = { ...options }; if (payload.all_steps) { payload.aggregate_all_job = ''; payload.gpu_step = ''; payload.npu_step = '' } job.value = await api(`/jobs/${job.value.id}/run`, 'POST', payload); await refresh() }) }
async function cancel() { await action(async () => { job.value = await api(`/jobs/${job.value.id}/cancel`, 'POST'); await refresh() }) }
async function removeJob() { if (!confirm('永久删除此任务的输入、所有运行结果和日志？此操作不可恢复。')) return; await action(async () => { await api(`/jobs/${job.value.id}?confirm=true`, 'DELETE'); job.value = null; localStorage.removeItem('profiling-job'); await refresh() }) }
function resultUrl(params) { return `/jobs/${job.value.id}/results?` + new URLSearchParams(Object.fromEntries(Object.entries(params).filter(([,v]) => v !== null && v !== undefined))) }
async function loadResults() {
  if (isPair.value || job.value?.status !== 'succeeded') return
  if (['operator','ai'].includes(tab.value)) return
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
  if(isPair.value||job.value?.status!=='succeeded'||['overview','operator','ai'].includes(tab.value)) return
  const id=job.value.id,sequence=++recordsSequence
  loadingRecords.value=true;records.value=[];recordTotal.value=0
  try {
    const result=await api(resultUrl({...recordParams(),offset:recordPage.value*20,limit:20}))
    if(job.value?.id!==id||sequence!==recordsSequence)return
    records.value=result.items;recordTotal.value=result.total
  } finally {if(sequence===recordsSequence)loadingRecords.value=false}
}
async function openType(row) { remember('类型对比');scope.value={op_type:row.op_type,device_id:row.device_id,core_type:row.core_type};tab.value='coverage';page.value=0;recordPage.value=0;recordKind.value='reasons';writeLocation(true);await action(loadResults) }
function openOperator(row) {remember(tabs.find(t=>t[0]===tab.value)?.[1]||'类型对比');scope.value={op_type:row.op_type,device_id:row.device_id,core_type:row.core_type};operatorState.value={...cleanView().operator,attempt:job.value.attempt};tab.value='operator';writeLocation(true)}
function updateOperator(state) {operatorState.value=cleanView({operator:state}).operator;writeLocation()}
async function returnTypes() {const index=trail.value.findLastIndex(item=>item.view.tab==='overview');if(index>=0)await backTo(index);else await switchTab('overview')}
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
const bars = computed(() => { const data = chartRows.value.slice(0, 12).reverse(); return { tooltip, legend: { data: ['NPU 总时间', '已分配 GPU 时间'], top: 0, left: 'center' }, grid: { left: 160, right: 35, top: 55, bottom: 50, containLabel: true }, xAxis: { type: 'value', name: 'μs', nameGap: 8 }, yAxis: { type: 'category', data: data.map(r => `${r.op_type} · ${r.device_id}/${r.core_type}`), axisLabel: { width: 145, overflow: 'truncate' } }, series: [{ name: 'NPU 总时间', type: 'bar', data: data.map(r => r.npu_total_time_us), itemStyle: { color: '#167d78' } }, { name: '已分配 GPU 时间', type: 'bar', data: data.map(r => r.gpu_total_time_us), itemStyle: { color: '#7489ba' } }] } })
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
    <aside class="workbench-sidebar"><div class="brand"><div class="brand-mark">P<span>↗</span></div><div>Profiling<span>WORKBENCH</span></div></div><div class="workspace-label">单机 · 共享工作区</div><button class="new-compare" :disabled="busy" @click="openCreate"><span>＋</span>新建比较</button><div class="section-label">任务 <span>{{ jobTotal }}</span></div><input v-model="jobSearch" placeholder="搜索任务名" aria-label="搜索任务名" @keyup.enter="findJobs"><div class="task-filters"><select v-model="jobStatus" @change="findJobs" aria-label="任务状态"><option value="">全部状态</option><option v-for="s in ['succeeded','running','queued','failed','needs_selection','interrupted','cancelled','uploading']" :key="s" :value="s">{{ labels[s] }}</option></select><label class="check"><input type="checkbox" v-model="jobArchived" @change="findJobs">归档</label><button @click="findJobs">搜索</button></div><nav class="task-list" aria-label="分析任务"><button v-for="j in jobs" :key="j.id" :class="['job-link', { selected: job?.id === j.id }]" @click="action(() => select(j.id))" :disabled="busy"><span class="job-name">{{ j.name }}</span><span class="job-status"><i :class="j.status"></i>{{ labels[j.status] }} · {{ new Date(j.created * 1000).toLocaleDateString() }}</span></button></nav><div class="pagination"><button :disabled="jobPage===0 || busy" @click="jobPage--;action(refresh)">上一页</button><span>{{ jobPage+1 }}</span><button :disabled="(jobPage+1)*30>=jobTotal || busy" @click="jobPage++;action(refresh)">下一页</button></div><footer><span :class="['worker-dot', { online: config.worker_online }]"></span>{{ config.worker_online ? 'Worker 在线 · 并发 1' : 'Worker 离线，请检查服务' }}<button class="text-button" v-if="config.auth_mode === 'token'" @click="logout">退出</button><span v-else>{{ config.auth_mode==='none'?'共享工作区 · 免令牌':'本机免令牌' }}</span></footer></aside>
    <main class="workbench-main"><header class="workbench-header"><div><p class="eyebrow">GPU / ASCEND NPU</p><form v-if="editingName" class="rename-form" @submit.prevent="renameJob"><input v-model="editedName" aria-label="任务名称" maxlength="120" required><button class="primary" :disabled="busy || !editedName.trim()">保存</button><button type="button" @click="editingName=false">取消</button></form><h1 v-else>{{ job?.name || '性能差异，从证据开始' }} <button v-if="job" class="text-button" @click="editedName=job.name;editingName=true">修改名称</button> <button v-if="job" class="text-button" :disabled="sharing" @click="shareCurrentView">{{ sharing ? '生成中…' : '分享链接' }}</button></h1><p class="muted">{{ job ? '任务 ' + job.id.slice(0, 8) + ' · ' + (labels[job.status] || job.status) : '上传两端采集数据，生成可追溯的对齐结果与覆盖率分析。' }}</p><p v-if="shareLink" class="share-row"><span>短链接</span><input :value="shareLink" readonly aria-label="分享短链接" @focus="$event.target.select()"><button type="button" @click="copyShare">{{ shareCopied ? '已复制' : '复制' }}</button><button type="button" class="text-button" @click="shareLink=''">收起</button></p></div><span class="privacy">● Profiling 本地处理 · AI 按需启用</span></header>
      <div class="service-toolbar"><button @click="showService">服务状态</button><span>{{ config.auth_mode==='token'?'令牌认证 · 共享工作区':config.auth_mode==='none'?'免令牌 · 共享工作区':'本机免令牌访问' }}</span></div><section v-if="service" class="panel"><h2>服务状态</h2><p>Worker {{ service.worker_online?'在线':'离线' }} · 排队 {{ service.tasks.queued || 0 }} · 运行 {{ service.tasks.running || 0 }}</p><p>存储 {{ filesize(service.storage_used_bytes) }} / {{ filesize(service.storage_bytes) }} · 内存软限制 {{ service.memory_mb }} MiB · 超时 {{ service.timeout_seconds }} 秒</p><p class="muted">状态为打开时快照。内存上限不是当前内存用量。</p></section><div v-if="error" class="error banner" role="alert">{{ error }}<button @click="error = ''" aria-label="关闭错误">×</button></div>
      <section v-if="!job" class="empty panel"><div class="empty-symbol">↗</div><h2>创建第一组性能比较</h2><p>GPU ↔ NPU 需要两侧 trace 和 NPU 明细；NPU ↔ NPU 只需两份 kernel_details.csv。</p><div class="empty-modes"><button @click="startNew('gpu_npu')"><b>GPU ↔ NPU</b><small>双侧实测 · 3 份输入</small></button><button @click="startNew('npu_pair')"><b>NPU ↔ NPU</b><small>前后对比 · 2 份明细</small></button></div><div class="steps"><span>01 选类型</span><span>02 上传数据</span><span>03 确认范围</span><span>04 分析与查证</span></div></section>
      <template v-else>
        <div class="toolbar"><span class="muted">输入和报告保存在服务器共享工作区；AI 分析经确认后发送所选证据。</span><button v-if="!active && job.status !== 'uploading'" :disabled="busy" @click="cloneJob">复用输入，新建分析</button><button v-if="!active" :disabled="busy || !!job.uploading" @click="archiveJob">{{ job.archived?'取消归档':'归档任务' }}</button><button v-if="!active" :disabled="busy || !!job.uploading" @click="removeJob">删除任务…</button></div>
        <section v-if="job.status === 'uploading'" class="panel"><div class="panel-heading"><h2>01 / 上传采集数据</h2><span class="muted">单文件上限 {{ filesize(config.max_upload_bytes) }}</span></div><div class="drop-zone" @dragover.prevent @drop.prevent="acceptFiles"><strong>拖拽多个文件到此处</strong><p class="muted">按文件名建议角色，请检查下方分配。无法识别的文件可手动选择角色。</p><input type="file" multiple accept=".json,.csv" aria-label="批量选择输入文件" :disabled="busy" @change="acceptFiles"></div><div v-for="(f,i) in unassignedFiles" :key="i" class="unassigned">{{ f.name }} <select aria-label="分配文件角色" @change="assignFile(i,$event.target.value)"><option value="">选择文件角色</option><option v-for="r in roles" :key="r.key" :value="r.key">{{ r.title }}</option></select><button @click="unassignedFiles.splice(i,1)">移除</button></div><div class="upload-grid"><label v-for="role in roles" :key="role.key" class="upload-box"><span class="file-icon">{{ role.accept === '.csv' ? 'CSV' : 'JSON' }}</span><strong>{{ role.title }}</strong><span class="muted">{{ role.hint }}</span><input type="file" :accept="role.accept" :disabled="busy" @change="files[role.key] = $event.target.files[0]"><span class="small">{{ files[role.key]?.name || job.inputs?.[role.key]?.name || '请选择文件' }}</span><span v-if="job.inputs?.[role.key]" class="success">已保存 · {{ filesize(job.inputs[role.key].size) }}</span></label></div><p class="muted">文件将流式保存到服务器；不会在浏览器解析完整 trace。输入开始预检查后冻结。</p><div v-if="uploading" class="upload-progress">{{ uploading }} · {{ uploadPercent }}%<progress :value="uploadPercent" max="100"></progress><button @click="cancelUpload">取消当前上传</button></div><button class="primary" :disabled="busy" @click="uploadAll">{{ busy ? '正在上传…' : '上传并预检查 →' }}</button></section>
        <section v-if="active" class="panel running"><div class="spinner"></div><div><h2>{{ labels[job.stage] || job.stage }}</h2><p class="muted">{{ job.status === 'queued' ? '任务已持久化排队，可安全刷新页面。' : `已运行约 ${elapsed} 秒。阶段进度不是预估百分比。` }}</p></div><button :disabled="busy || !!job.cancel" @click="cancel">{{ job.cancel ? '正在取消…' : '取消任务' }}</button></section>
        <div v-if="job.error" class="error banner"><strong>{{ job.error }}</strong><p>{{ errorAdvice }}</p></div>
        <NpuPairPanel v-if="isPair && !active" :key="job.id" :job="job" :api="api" :busy="busy" @run="opts => action(async () => { job = await api('/jobs/'+job.id+'/run-npu-pair', 'POST', opts); await refresh() })" />
        <section v-if="!isPair && job.inspection && job.status !== 'succeeded' && !active" class="panel"><div class="panel-heading"><h2>02 / 确认比较范围</h2><span class="tag">{{ job.inspection.status }}</span></div><ul class="warnings"><li v-for="(w, i) in [...job.inspection.errors, ...job.inspection.warnings, ...job.inspection.selections_required]" :key="i">{{ w.side }} · {{ w.message }}</li></ul><div class="scope-grid"><div v-for="side in ['gpu', 'npu']" :key="side"><h3>{{ side.toUpperCase() }}</h3><label>CPU PID<select v-model="options[side + '_pid']" @change="options[side + '_step'] = ''"><option value="">选择进程</option><option v-for="c in candidates(side)" :key="c.pid" :value="c.pid">{{ c.pid }} · {{ c.op_count }} CPU ops</option></select></label><label>Step<select v-model="options[side + '_step']" :disabled="options.all_steps"><option value="">选择稳定 step</option><option v-for="(s, i) in steps(side)" :key="i" :value="normalizeStep(s.name)">{{ s.name }} · {{ fmt(s.duration_us) }} μs</option></select></label></div></div><div class="options"><label class="check"><input type="checkbox" v-model="options.all_steps">明确比较完整采集范围（包括 warmup / 未标记事件）</label><label class="check"><input type="checkbox" v-model="options.include_communication">包含通信 kernel</label><label class="check"><input type="checkbox" v-model="options.export_call_tables">额外生成严格调用评审 Excel（增加分析时间）</label><label v-if="!options.all_steps">类型＋调用方聚合表（可选全量任务）<select v-model="options.aggregate_all_job"><option value="">不生成聚合表</option><option v-for="j in aggregateSources" :key="j.id" :value="j.id">{{ j.name }} · {{ j.id.slice(0,8) }}</option></select><small>仅列出同一组输入的已完成全量任务。</small></label><label>诊断阈值 %<input type="number" v-model.number="options.diagnostic_coverage_threshold" min="0" max="100"></label></div><p class="notice">相同 step 编号不证明输入和工作量等价。请确认 batch、shape、精度及设备范围一致；PID 不是设备过滤器。</p><button class="primary" :disabled="busy || !selectable || job.inspection.status === 'invalid'" @click="run">开始分析 →</button></section>
        <section v-if="['failed', 'cancelled', 'interrupted'].includes(job.status) && !job.inspection" class="panel"><p>可重试预检查。若原始输入有误，请新建任务上传修正后的文件。</p><button @click="action(async () => { job = await api(`/jobs/${job.id}/inspect`, 'POST') })">重试预检查</button></section>
        <template v-if="job.status === 'succeeded' && summary && !isPair">
          <p class="notice">全任务比较范围（不随下方筛选改变）：GPU PID {{ job.options.gpu_pid || '自动选择' }} / {{ job.options.all_steps ? '完整采集' : job.options.gpu_step }}；NPU PID {{ job.options.npu_pid || '自动选择' }} / {{ job.options.all_steps ? '完整采集' : job.options.npu_step }}。耗时图沿用类型表的当前筛选条件。</p>
          <ReportDownloads :job="job"><button @click="openAI()">AI 辅助分析</button><span class="tag success">✓ {{ summary.saved_validation?.checks_run }} 项校验通过</span><span class="muted">引擎 {{ summary.producer?.version }} · {{ summary.status === 'complete_with_gaps' ? '已完成，仍有覆盖缺口' : '已完成' }}</span></ReportDownloads>
          <div v-if="!['operator','ai'].includes(tab)" class="metrics"><article><span>NPU 总时间</span><strong>{{ fmt(overall?.npu_total_time_us) }}<small> μs</small></strong><p>kernel_details-backed</p></article><article><span>GPU 总时间</span><strong>{{ fmt(overall?.gpu_total_time_us) }}<small> μs</small></strong><p>计算 kernel + 内存活动，去重</p></article><article><span>Comparable Ratio · GPU/NPU</span><strong :class="ratioClass(overall?.comparable_ratio_gpu_npu)">{{ fmt(overall?.comparable_ratio_gpu_npu, 3) }}<small> ×</small></strong><p>对应可比子集，不包含所有工作</p></article><article><span>Total Time Ratio · GPU/NPU</span><strong :class="ratioClass(overall?.total_time_ratio_gpu_npu)">{{ fmt(overall?.total_time_ratio_gpu_npu, 3) }}<small> ×</small></strong><p>总体范围比较，非等价工作量证明</p></article></div>
          <CoreBreakdown v-if="!['operator','ai'].includes(tab)" :data="summary.core_breakdown" />
          <ParameterSummary v-if="!['operator','ai'].includes(tab)" :data="summary.parameters" />
          <details class="notice"><summary>统计口径与 {{ summary.warnings.length }} 条警告</summary><p>耗时是事件时长之和，不是墙钟时间。NPU trace-only 任务不计入 TOTAL；GPU 未映射计算与内存活动计入。整体算子视图与类型视图重叠，不能累加。</p><p v-for="(w, i) in summary.warnings" :key="i">{{ w.message }} <span v-if="w.count">数量 {{ w.count }}，{{ fmt(w.time_us) }} μs</span></p></details>
          <div class="tabs" role="tablist"><button v-for="[key, label] in tabs" :key="key" :class="{ active: tab === key }" @click="switchTab(key)" role="tab" :aria-selected="tab === key">{{ label }}</button></div>
          <section class="panel result-panel">
            <nav v-if="trail.length" class="breadcrumbs" aria-label="分析路径"><button v-for="(item,i) in trail" :key="i" @click="backTo(i)">{{ item.label }} ←</button><span>{{ tabs.find(t=>t[0]===tab)?.[1] }}</span></nav>
            <div v-if="tab!=='ai'" class="toolbar"><label class="check"><input type="checkbox" v-model="detailed">详细模式</label><select v-model="savedViewIndex" @change="useView" aria-label="已保存视图"><option value="">选择已保存视图</option><option v-for="(v,i) in savedViews" :key="i" :value="String(i)">{{ v.name }}</option></select><button @click="savingView=!savingView">保存视图</button><form v-if="savingView" class="toolbar" @submit.prevent="saveView"><input v-model="viewName" aria-label="视图名称" placeholder="输入视图名称" maxlength="60" required><button type="submit">确认保存视图</button><button type="button" @click="savingView=false">取消</button></form><button v-if="savedViewIndex!==''" @click="deleteView">删除视图</button><button v-if="!['operator','ai'].includes(tab)" :disabled="busy" @click="exportView">导出当前筛选 CSV</button></div>
            <p class="muted small">视图设置只影响展示。分析范围和诊断生成阈值的变更需“复用输入，新建分析”。</p>
            <p v-if="loadingResults || loadingRecords" role="status">正在加载结果…</p>
            <AiAssistant v-if="tab==='ai'" :key="job.id+JSON.stringify(aiScope)" :job="job" :scope="aiScope" :api="api" @scope="openAI" @associate="aiAssociate"/>
            <OperatorDetails v-if="tab==='operator'" :key="job.id+job.attempt" :job="job" :scope="scope" :state="operatorState" :api="api" :detailed="detailed" @update="updateOperator" @whole="openWhole" @coverage="openType" @upgrade="cloneJob" @types="returnTypes" @ai="openAI"/>
            <template v-if="tab === 'overview' || tab === 'coverage'"><div class="toolbar"><input v-model="search" placeholder="搜索 OP Type" @keyup.enter="filter" aria-label="搜索类型"><select v-model="sort" @change="filter" aria-label="排序"><option value="npu_time">NPU 耗时降序</option><option value="coverage">可比时间覆盖率升序</option><option value="ratio">可比性能比降序</option><option value="name">类型名称</option></select><label class="check"><input type="checkbox" v-model="hotspot" @change="filter">NPU 占比 &gt; 1% 且可比时间覆盖率 &lt; 90%</label><button @click="filter">筛选</button></div><Chart v-if="tab === 'coverage'" :option="scatter" @select="clickedChart"/><p v-if="tab === 'coverage'" class="muted small">气泡大小表示不可比较耗时，不代表可节省时间。图表最多展示筛选后前 200 个类型。</p><p class="filter-context">当前类型范围：{{ scope ? scope.op_type+' / '+scope.device_id+' / '+scope.core_type : '全部类型' }} · 搜索 {{ search || '不限' }} · {{ hotspot?'高耗时低覆盖率':'不限覆盖率' }} <button @click="clearFilters">清除筛选</button></p><TypeTable v-if="!loadingResults" :rows="rows" :detailed="detailed" @type="openType" @detail="openOperator" @whole="openWhole" @report="openReport" @manual="openManual" @ai="openAI"/><div class="pagination"><span>共 {{ total }} 行 · 第 {{ page + 1 }} 页</span><button :disabled="page === 0" @click="page--; action(loadResults)">上一页</button><button :disabled="(page + 1) * 30 >= total" @click="page++; action(loadResults)">下一页</button></div></template>
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
            <template v-if="tab !== 'overview' && tab !== 'operator' && tab !== 'ai'"><div v-if="!loadingRecords && tab !== 'whole' && tab !== 'calls'" class="record-list"><article v-for="(r, i) in records" :key="i" class="record"><template v-if="tab === 'coverage' && recordKind === 'reasons'"><div class="record-heading"><strong>{{ r.op_type }} · {{ r.reason }}</strong><span>{{ r.kernel_count }} 个 · {{ fmt(r.npu_time_us) }} μs</span></div><p>{{ r.hypothesis }}</p><p class="small muted">CPU {{ r.cpu_op || '未知' }} · API {{ r.npu_api || '未知' }} · NPU 样本 {{ r.sample_kernel_indices?.join(', ') }}</p><ol><li v-for="step in r.next_steps" :key="step">{{ step }}</li></ol></template><template v-else><button class="type-link" @click="showRecord(r)">{{ r['Fused OP'] || r['CPU Bundle'] || r['NPU Kernel Name'] || r['GPU Name'] || r.op_type || '事件' }} <span v-if="r['NPU Kernel Index'] !== undefined">#{{ r['NPU Kernel Index'] }}</span></button><p v-if="r.cpu_alignment_status">{{ r.cpu_alignment_status }} · {{ r.kernel_comparison_status }}</p><p class="muted small">{{ r.source || r['Mapping Mode'] || r.reason || r['Status'] || '查看结构化证据' }}</p><button @click="showRecord(r)">查看详情 →</button></template></article><p v-if="!records.length" class="muted">没有对应记录。无诊断可能是类型未命中阈值，不代表已证明等价。</p></div><div class="pagination"><span>共 {{ recordTotal }} 条 · 第 {{ recordPage + 1 }} 页</span><button :disabled="recordPage === 0" @click="recordPage--; action(loadRecords)">上一页</button><button :disabled="(recordPage + 1) * 20 >= recordTotal" @click="recordPage++; action(loadRecords)">下一页</button></div></template>
          </section>
        </template>
      </template>
      <footer class="main-footer">PROFILING WORKBENCH <span>证据优先 · 独立计时 · 可追溯结果</span></footer>
    </main>
    <EvidenceDrawer v-if="detail" :record="detail" :evidence="evidence" @close="closeDetail"/>

    <ManualLinkDialog v-if="manualType" :key="job.id" :job="job" :type="manualType" :seed="manualSeed" :api="api" @close="manualType=null" @submitted="manualSubmitted" @upgrade="manualUpgrade"/>

    <dialog ref="createDialog" class="create-dialog" aria-labelledby="create-title" @cancel.prevent="closeCreate()">
      <form @submit.prevent="createJob">
        <header class="create-head">
          <p class="eyebrow">NEW COMPARISON</p>
          <h2 id="create-title">新建比较</h2>
          <p class="muted small">两种比较的输入与口径不同，创建后不能切换；需要换类型请新建任务。</p>
          <button type="button" class="create-close" aria-label="关闭" @click="closeCreate()">×</button>
        </header>
        <div class="create-body">
          <div class="mode-cards" role="radiogroup" aria-label="比较类型">
            <label v-for="m in modes" :key="m.id" :class="['mode-card', { selected: newMode === m.id }]">
              <input type="radio" name="compare-mode" :value="m.id" v-model="newMode">
              <span class="mode-glyph"><b>{{ m.from }}</b><i>⇄</i><b>{{ m.to }}</b></span>
              <span class="mode-title">{{ m.title }}<em>{{ m.tag }}</em></span>
              <span class="mode-summary">{{ m.summary }}</span>
              <span class="mode-inputs"><i v-for="i in m.inputs" :key="i">{{ i }}</i></span>
            </label>
          </div>
          <label class="create-name">任务名称（可选）
            <input v-model="name" maxlength="120" autofocus :placeholder="currentMode.placeholder" aria-label="比较任务名称">
          </label>
        </div>
        <footer class="create-actions">
          <span class="muted small">{{ currentMode.hint }}</span>
          <button type="button" @click="closeCreate()">取消</button>
          <button class="primary" :disabled="busy">{{ busy ? '正在创建…' : '创建并上传 →' }}</button>
        </footer>
      </form>
    </dialog>
  </div>
</template>
