<script setup>
import { computed, onMounted, onBeforeUnmount, ref, reactive, watch } from 'vue'
import Chart from './Chart.vue'
const logged = ref(false), token = ref(''), error = ref(''), busy = ref(false), config = ref({})
const jobs = ref([]), job = ref(null), tab = ref('overview'), name = ref(''), files = reactive({}), uploading = ref(''), uploadPercent = ref(0)
const rows = ref([]), chartRows = ref([]), total = ref(0), page = ref(0), search = ref(''), sort = ref('npu_time'), hotspot = ref(false)
const detail = ref(null), evidence = ref(null), scope = ref(null), records = ref([]), recordTotal = ref(0), recordPage = ref(0), recordKind = ref('reasons')
const options = reactive({ gpu_pid: '', npu_pid: '', gpu_step: '', npu_step: '', all_steps: false, diagnostic_coverage_threshold: 90, include_communication: false })
const roles = [{ key: 'gpu_trace', title: 'GPU trace', hint: 'PyTorch profiler · JSON', accept: '.json' }, { key: 'npu_trace', title: 'NPU trace', hint: 'Ascend trace_view · JSON', accept: '.json' }, { key: 'npu_kernel_details', title: 'NPU kernel details', hint: 'kernel_details · CSV', accept: '.csv' }]
const labels = { uploading: '等待上传', queued: '排队中', running: '分析中', needs_selection: '等待确认范围', succeeded: '分析完成', failed: '失败', cancelled: '已取消', interrupted: '已中断', inspection: '输入预检查', gpu_parse: '解析 GPU trace', npu_parse: '解析 NPU trace', detail_link: '关联 NPU 明细', matching: '匹配 CPU / kernel', reports: '生成报告', validation: '校验计时与映射', indexing: '构建查询索引', starting: '启动分析进程' }
const tabs = [['overview', '总览'], ['coverage', '覆盖率诊断'], ['distribution', '耗时分布'], ['calls', '整体算子']]
const summary = computed(() => job.value?.summary)
const overall = computed(() => summary.value?.overall_statistics?.[0])
const active = computed(() => ['running', 'queued'].includes(job.value?.status))
const selectable = computed(() => job.value?.inspection && ['needs_selection', 'failed', 'cancelled', 'interrupted'].includes(job.value?.status))
const elapsed = ref(0)
let timer, loading = false, selectionSequence = 0, resultsSequence = 0, recordsSequence = 0
const fmt = (v, n = 2) => v == null ? '—' : Number(v).toLocaleString('en-US', { maximumFractionDigits: n })
const ratioClass = v => v == null ? '' : v >= 2 ? 'ratio-dark' : v >= 1 ? 'ratio-light' : 'ratio-yellow'
const filesize = v => `${fmt(v / 1024 ** 2, 1)} MB`
async function api(path, method = 'GET', body) {
  const response = await fetch('/api' + path, { method, headers: body === undefined ? {} : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
  if (response.status === 401) logged.value = false
  if (!response.ok) { const e = await response.json().catch(() => ({})); throw new Error(typeof e.detail === 'string' ? e.detail : JSON.stringify(e.detail || response.statusText)) }
  return response.json()
}
async function action(fn) { error.value = ''; busy.value = true; try { await fn() } catch (e) { error.value = e.message } finally { busy.value = false } }
async function login() { await action(async () => { await api('/session', 'POST', { token: token.value }); token.value = ''; logged.value = true; await refresh(); const saved = localStorage.getItem('profiling-job'); if (saved && jobs.value.some(j => j.id === saved)) await select(saved) }) }
async function logout() { await action(async () => { await api('/session', 'DELETE'); logged.value = false; job.value = null }) }
async function refresh() { config.value = await api('/config'); jobs.value = (await api('/jobs')).items }
async function select(id) {
  const sequence = ++selectionSequence
  const next = await api('/jobs/' + id)
  if (sequence !== selectionSequence) return
  job.value = next; localStorage.setItem('profiling-job', id); page.value = 0; recordPage.value = 0; scope.value = null; detail.value = null; evidence.value = null; records.value = []; rows.value = []; chartRows.value = []
  Object.keys(files).forEach(k => delete files[k]); search.value = ''; hotspot.value = false
  Object.assign(options, { gpu_pid: '', npu_pid: '', gpu_step: '', npu_step: '', all_steps: false }, next.options || {})
  for (const side of ['gpu', 'npu']) { const list = candidates(side); if (!options[side + '_pid'] && list.length === 1) options[side + '_pid'] = list[0].pid }
  if (next.status === 'succeeded') await loadResults()
}
function candidates(side) { return job.value?.inspection?.profiles?.[side]?.cpu_candidates || [] }
function steps(side) { return candidates(side).find(c => c.pid === options[side + '_pid'])?.steps || [] }
async function createJob() { await action(async () => { const j = await api('/jobs', 'POST', { name: name.value.trim() || 'GPU / NPU comparison' }); name.value = ''; await refresh(); await select(j.id) }) }
function sendFile(id, role, file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest(); xhr.open('PUT', `/api/jobs/${id}/inputs/${role}`); xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name)); xhr.setRequestHeader('Content-Type', 'application/octet-stream')
    xhr.upload.onprogress = e => { uploadPercent.value = e.lengthComputable ? Math.round(e.loaded / e.total * 100) : 0 }
    xhr.onload = () => { if (xhr.status >= 200 && xhr.status < 300) resolve(); else { let message = '上传失败'; try { message = JSON.parse(xhr.responseText).detail } catch {} reject(new Error(message)) } }
    xhr.onerror = () => reject(new Error('上传连接中断，可以重新上传该文件')); xhr.send(file)
  })
}
async function uploadAll() { await action(async () => {
  const id = job.value.id
  try { for (const role of roles) { const file = files[role.key]; if (!file) { if (!job.value.inputs?.[role.key]) throw new Error('请提供 ' + role.title); continue } if (file.size > config.value.max_upload_bytes) throw new Error(role.title + ' 超过上传上限'); uploading.value = role.title; uploadPercent.value = 0; await sendFile(id, role.key, file) } await api(`/jobs/${id}/inspect`, 'POST'); await select(id); await refresh() } finally { uploading.value = '' }
}) }
async function run() { await action(async () => { const payload = { ...options }; if (payload.all_steps) { payload.gpu_step = ''; payload.npu_step = '' } job.value = await api(`/jobs/${job.value.id}/run`, 'POST', payload); await refresh() }) }
async function cancel() { await action(async () => { job.value = await api(`/jobs/${job.value.id}/cancel`, 'POST'); await refresh() }) }
async function removeJob() { if (!confirm('永久删除此任务的输入、所有运行结果和日志？此操作不可恢复。')) return; await action(async () => { await api(`/jobs/${job.value.id}?confirm=true`, 'DELETE'); job.value = null; localStorage.removeItem('profiling-job'); await refresh() }) }
function resultUrl(params) { return `/jobs/${job.value.id}/results?` + new URLSearchParams(Object.fromEntries(Object.entries(params).filter(([,v]) => v !== null && v !== undefined))) }
async function loadResults() {
  if (job.value?.status !== 'succeeded') return
  const id = job.value.id
  const sequence = ++resultsSequence
  const params = { search: search.value, sort: sort.value, limit: 30, offset: page.value * 30, ...(hotspot.value ? { coverage_max: 90, npu_ratio_min: 1 } : {}) }
  const [data, chart] = await Promise.all([api(resultUrl(params)), api(resultUrl({ ...params, sort: 'npu_time', limit: 200, offset: 0 }))])
  if (job.value?.id !== id || sequence !== resultsSequence) return
  rows.value = data.items; total.value = data.total; chartRows.value = chart.items.filter(x => x.row_kind === 'npu_type')
  if (['coverage', 'calls', 'distribution'].includes(tab.value)) await loadRecords()
}
async function loadRecords() {
  if (job.value?.status !== 'succeeded') return
  const id = job.value.id
  const sequence = ++recordsSequence
  const kind = tab.value === 'calls' ? 'calls' : tab.value === 'distribution' ? 'gpu_unmapped' : recordKind.value
  const filters = tab.value === 'coverage' && scope.value ? scope.value : {}
  const result = await api(resultUrl({ kind, ...filters, offset: recordPage.value * 20, limit: 20 }))
  if (job.value?.id !== id || sequence !== recordsSequence) return
  records.value = result.items; recordTotal.value = result.total
}
async function openType(row) { scope.value = { op_type: row.op_type, device_id: row.device_id, core_type: row.core_type }; tab.value = 'coverage'; recordPage.value = 0; recordKind.value = 'reasons'; await action(loadRecords) }
async function showRecord(record) {
  detail.value = record; evidence.value = null
  const index = record['NPU Kernel Index'] ?? record.npu_kernel_index
  if (index !== undefined) await action(async () => { evidence.value = await api(resultUrl({ kind: 'diagnoses', npu_index: index, limit: 5 })) })
}
function download(name) { return `/api/jobs/${job.value.id}/artifacts/${encodeURIComponent(name)}` }
async function switchTab(value) { tab.value = value; recordPage.value = 0; await action(loadRecords) }
async function filter() { page.value = 0; await action(loadResults) }
function clickedChart(p) { const row = chartRows.value.find(x => `${x.op_type} · ${x.device_id}/${x.core_type}` === p.name); if (row) openType(row) }
const tooltip = { trigger: 'axis', renderMode: 'richText', confine: true }
const bars = computed(() => { const data = chartRows.value.slice(0, 12).reverse(); return { tooltip, legend: { data: ['NPU 总时间', '已分配 GPU 时间'] }, grid: { left: 160, right: 30, top: 40, bottom: 35 }, xAxis: { type: 'value', name: 'μs' }, yAxis: { type: 'category', data: data.map(r => `${r.op_type} · ${r.device_id}/${r.core_type}`), axisLabel: { width: 145, overflow: 'truncate' } }, series: [{ name: 'NPU 总时间', type: 'bar', data: data.map(r => r.npu_total_time_us), itemStyle: { color: '#167d78' } }, { name: '已分配 GPU 时间', type: 'bar', data: data.map(r => r.gpu_total_time_us), itemStyle: { color: '#7489ba' } }] } })
const scatter = computed(() => ({ tooltip: { renderMode: 'richText', confine: true }, grid: { left: 60, right: 40, top: 20, bottom: 50 }, xAxis: { name: 'NPU 时间占比 %', nameLocation: 'middle', nameGap: 30, type: 'value' }, yAxis: { name: 'Comparable 时间覆盖率 %', max: 100, min: 0 }, series: [{ type: 'scatter', itemStyle: { color: '#167d78', opacity: .7 }, data: chartRows.value.map(r => ({ name: `${r.op_type} · ${r.device_id}/${r.core_type}`, value: [r.npu_ratio_pct, r.coverage.comparable.time_pct], symbolSize: Math.min(32, 8 + Math.sqrt(Math.max(0, r.npu_total_time_us - r.npu_comparable_time_us))) })) }] }))
watch(recordKind, () => { recordPage.value = 0; action(loadRecords) })
onMounted(async () => {
  try { await refresh(); logged.value = true; const saved = localStorage.getItem('profiling-job'); if (saved && jobs.value.some(j => j.id === saved)) await select(saved) } catch {}
  timer = setInterval(async () => { if (!logged.value || loading || busy.value) return; loading = true; try { await refresh(); if (job.value) { const before = job.value.status; const id = job.value.id; const next = await api('/jobs/' + id); if (job.value?.id === id) { job.value = next; elapsed.value = next.started ? Math.round((Date.now() / 1000 - next.started)) : 0; if (before !== next.status) { if (next.status === 'succeeded') await loadResults(); if (next.status === 'needs_selection') for (const side of ['gpu', 'npu']) if (candidates(side).length === 1) options[side + '_pid'] = candidates(side)[0].pid } } } } catch(e) { error.value = e.message } finally { loading = false } }, 2500)
})
onBeforeUnmount(() => clearInterval(timer))
</script>

<template>
  <div v-if="!logged" class="login-page"><section class="login-card"><div class="brand-mark">P<span>↗</span></div><p class="eyebrow">LOCAL ANALYSIS WORKSPACE</p><h1>Profiling Workbench</h1><p class="muted">把 GPU 与 NPU 的性能差异，追溯到每一次调用。</p><form @submit.prevent="login"><label>访问令牌<input v-model="token" type="password" autocomplete="current-password" placeholder="粘贴服务器访问令牌" required></label><button class="primary" :disabled="busy">进入工作台 →</button></form><p class="muted small">令牌位于服务器数据目录的 access-token 文件，或由管理员提供。此版本是单机共享工作区。</p><p v-if="error" class="error">{{ error }}</p></section></div>
  <div v-else class="app-shell">
    <aside><div class="brand"><div class="brand-mark">P<span>↗</span></div><div>Profiling<span>WORKBENCH</span></div></div><div class="workspace-label">单机 · 共享工作区</div><form class="new-job" @submit.prevent="createJob"><input v-model="name" placeholder="比较任务名称" maxlength="120" aria-label="新任务名称"><button class="primary" :disabled="busy">＋ 新建比较</button></form><div class="section-label">最近任务 <span>{{ jobs.length }}</span></div><nav><button v-for="j in jobs" :key="j.id" :class="['job-link', { selected: job?.id === j.id }]" @click="action(() => select(j.id))" :disabled="busy"><span class="job-name">{{ j.name }}</span><span class="job-status"><i :class="j.status"></i>{{ labels[j.status] }} · {{ new Date(j.created * 1000).toLocaleDateString() }}</span></button></nav><footer><span :class="['worker-dot', { online: config.worker_online }]"></span>{{ config.worker_online ? 'Worker 在线 · 并发 1' : 'Worker 离线，请检查服务' }}<button class="text-button" @click="logout">退出</button></footer></aside>
    <main><header><div><p class="eyebrow">GPU / ASCEND NPU</p><h1>{{ job?.name || '性能差异，从证据开始' }}</h1><p class="muted">{{ job ? '任务 ' + job.id.slice(0, 8) + ' · ' + (labels[job.status] || job.status) : '上传两端采集数据，生成可追溯的对齐结果与覆盖率分析。' }}</p></div><span class="privacy">● 本地处理 · 无外部 AI 调用</span></header>
      <div v-if="error" class="error banner" role="alert">{{ error }}<button @click="error = ''" aria-label="关闭错误">×</button></div>
      <section v-if="!job" class="empty panel"><div class="empty-symbol">↗</div><h2>创建第一组性能比较</h2><p>准备 GPU trace、NPU trace 和 kernel_details.csv。<br>在左侧新建任务，选择比较范围后开始分析。</p><div class="steps"><span>01 上传数据</span><span>02 确认范围</span><span>03 分析与查证</span></div></section>
      <template v-else>
        <div class="toolbar"><span class="muted">输入和输出仅保存在此服务器的共享工作区。</span><button v-if="!active" :disabled="busy || !!job.uploading" @click="removeJob">删除任务…</button></div>
        <section v-if="job.status === 'uploading'" class="panel"><div class="panel-heading"><h2>01 / 上传采集数据</h2><span class="muted">单文件上限 {{ filesize(config.max_upload_bytes) }}</span></div><div class="upload-grid"><label v-for="role in roles" :key="role.key" class="upload-box"><span class="file-icon">{{ role.accept === '.csv' ? 'CSV' : 'JSON' }}</span><strong>{{ role.title }}</strong><span class="muted">{{ role.hint }}</span><input type="file" :accept="role.accept" :disabled="busy" @change="files[role.key] = $event.target.files[0]"><span class="small">{{ files[role.key]?.name || job.inputs?.[role.key]?.name || '请选择文件' }}</span><span v-if="job.inputs?.[role.key]" class="success">已保存 · {{ filesize(job.inputs[role.key].size) }}</span></label></div><p class="muted">文件将流式保存到服务器；不会在浏览器解析完整 trace。输入开始预检查后冻结。</p><div v-if="uploading" class="upload-progress">{{ uploading }} · {{ uploadPercent }}%<progress :value="uploadPercent" max="100"></progress></div><button class="primary" :disabled="busy" @click="uploadAll">{{ busy ? '正在上传…' : '上传并预检查 →' }}</button></section>
        <section v-if="active" class="panel running"><div class="spinner"></div><div><h2>{{ labels[job.stage] || job.stage }}</h2><p class="muted">{{ job.status === 'queued' ? '任务已持久化排队，可安全刷新页面。' : `已运行约 ${elapsed} 秒。阶段进度不是预估百分比。` }}</p></div><button :disabled="busy || !!job.cancel" @click="cancel">{{ job.cancel ? '正在取消…' : '取消任务' }}</button></section>
        <div v-if="job.error" class="error banner">{{ job.error }}</div>
        <section v-if="job.inspection && job.status !== 'succeeded' && !active" class="panel"><div class="panel-heading"><h2>02 / 确认比较范围</h2><span class="tag">{{ job.inspection.status }}</span></div><ul class="warnings"><li v-for="(w, i) in [...job.inspection.errors, ...job.inspection.warnings, ...job.inspection.selections_required]" :key="i">{{ w.side }} · {{ w.message }}</li></ul><div class="scope-grid"><div v-for="side in ['gpu', 'npu']" :key="side"><h3>{{ side.toUpperCase() }}</h3><label>CPU PID<select v-model="options[side + '_pid']" @change="options[side + '_step'] = ''"><option value="">选择进程</option><option v-for="c in candidates(side)" :key="c.pid" :value="c.pid">{{ c.pid }} · {{ c.op_count }} CPU ops</option></select></label><label>Step<select v-model="options[side + '_step']" :disabled="options.all_steps"><option value="">选择稳定 step</option><option v-for="(s, i) in steps(side)" :key="i" :value="s.name">{{ s.name }} · {{ fmt(s.duration_us) }} μs</option></select></label></div></div><div class="options"><label class="check"><input type="checkbox" v-model="options.all_steps">明确比较完整采集范围（包括 warmup / 未标记事件）</label><label class="check"><input type="checkbox" v-model="options.include_communication">包含通信 kernel</label><label>诊断阈值 %<input type="number" v-model.number="options.diagnostic_coverage_threshold" min="0" max="100"></label></div><p class="notice">相同 step 编号不证明输入和工作量等价。请确认 batch、shape、精度及设备范围一致；PID 不是设备过滤器。</p><button class="primary" :disabled="busy || !selectable || job.inspection.status === 'invalid'" @click="run">开始分析 →</button></section>
        <section v-if="['failed', 'cancelled', 'interrupted'].includes(job.status) && !job.inspection" class="panel"><p>可重试预检查。若原始输入有误，请新建任务上传修正后的文件。</p><button @click="action(async () => { job = await api(`/jobs/${job.id}/inspect`, 'POST') })">重试预检查</button></section>
        <template v-if="job.status === 'succeeded' && summary">
          <p class="notice">比较范围：GPU PID {{ job.options.gpu_pid || '自动选择' }} / {{ job.options.all_steps ? '完整采集' : job.options.gpu_step }}；NPU PID {{ job.options.npu_pid || '自动选择' }} / {{ job.options.all_steps ? '完整采集' : job.options.npu_step }}。耗时图沿用类型表的当前筛选条件。</p>
          <div class="result-meta"><span class="tag success">✓ {{ summary.saved_validation?.checks_run }} 项校验通过</span><span class="muted">引擎 {{ summary.producer?.version }} · {{ summary.status === 'complete_with_gaps' ? '已完成，仍有覆盖缺口' : '已完成' }}</span><a class="button primary" :href="download('npu_gpu_op_statistic.xlsx')">↓ 下载 Excel</a><details class="downloads"><summary>更多输出</summary><a v-for="file in summary.reports" :key="file" :href="download(file)">{{ file }}</a></details></div>
          <div class="metrics"><article><span>NPU 总时间</span><strong>{{ fmt(overall?.npu_total_time_us) }}<small> μs</small></strong><p>kernel_details-backed</p></article><article><span>GPU 总时间</span><strong>{{ fmt(overall?.gpu_total_time_us) }}<small> μs</small></strong><p>计算 kernel + 内存活动，去重</p></article><article><span>Comparable Ratio · GPU/NPU</span><strong :class="ratioClass(overall?.comparable_ratio_gpu_npu)">{{ fmt(overall?.comparable_ratio_gpu_npu, 3) }}<small> ×</small></strong><p>对应可比子集，不包含所有工作</p></article><article><span>Total Time Ratio · GPU/NPU</span><strong :class="ratioClass(overall?.total_time_ratio_gpu_npu)">{{ fmt(overall?.total_time_ratio_gpu_npu, 3) }}<small> ×</small></strong><p>总体范围比较，非等价工作量证明</p></article></div>
          <details class="notice"><summary>统计口径与 {{ summary.warnings.length }} 条警告</summary><p>耗时是事件时长之和，不是墙钟时间。NPU trace-only 任务不计入 TOTAL；GPU 未映射计算与内存活动计入。整体算子视图与类型视图重叠，不能累加。</p><p v-for="(w, i) in summary.warnings" :key="i">{{ w.message }} <span v-if="w.count">数量 {{ w.count }}，{{ fmt(w.time_us) }} μs</span></p></details>
          <div class="tabs" role="tablist"><button v-for="[key, label] in tabs" :key="key" :class="{ active: tab === key }" @click="switchTab(key)" role="tab" :aria-selected="tab === key">{{ label }}</button></div>
          <section class="panel result-panel">
            <template v-if="tab === 'overview' || tab === 'coverage'"><div class="toolbar"><input v-model="search" placeholder="搜索 OP Type" @keyup.enter="filter" aria-label="搜索类型"><select v-model="sort" @change="filter" aria-label="排序"><option value="npu_time">NPU 耗时降序</option><option value="coverage">可比时间覆盖率升序</option><option value="ratio">可比性能比降序</option><option value="name">类型名称</option></select><label class="check"><input type="checkbox" v-model="hotspot" @change="filter">NPU 占比 &gt; 1% 且可比时间覆盖率 &lt; 90%</label><button @click="filter">筛选</button></div><Chart v-if="tab === 'coverage'" :option="scatter" @select="clickedChart"/><p v-if="tab === 'coverage'" class="muted small">气泡大小表示不可比较耗时，不代表可节省时间。图表最多展示筛选后前 200 个类型。</p><div class="table-scroll"><table><thead><tr><th>OP Type / 融合分组</th><th>NPU / GPU 数量</th><th>NPU / GPU 总时间 μs</th><th>Comparable Ratio</th><th>Total Time Ratio</th><th>覆盖率 %<small>数量 / 时间</small></th><th>排查结论</th></tr></thead><tbody><tr v-for="r in rows" :key="r.device_id + r.op_type + r.core_type"><td><button class="type-link" @click="openType(r)" :disabled="r.row_kind !== 'npu_type'">{{ r.op_type }}</button><small>{{ r.device_id }} / {{ r.core_type }}</small><small v-if="r.fused_npu_type !== r.op_type || r.fused_gpu_type !== r.op_type">{{ r.fused_npu_type }} ↔ {{ r.fused_gpu_type }}</small></td><td>{{ fmt(r.npu_kernel_count, 0) }} / {{ fmt(r.gpu_kernel_count, 0) }}<small>GPU 内存活动 {{ fmt(r.gpu_memory_activity_count, 0) }}</small></td><td>{{ fmt(r.npu_total_time_us) }}<small>{{ fmt(r.gpu_total_time_us) }}</small></td><td><span :class="['ratio', ratioClass(r.comparable_ratio_gpu_npu)]">{{ fmt(r.comparable_ratio_gpu_npu, 3) }}</span></td><td><span :class="['ratio', ratioClass(r.total_time_ratio_gpu_npu)]">{{ fmt(r.total_time_ratio_gpu_npu, 3) }}</span></td><td><div v-for="k in ['direct', 'comparable', 'accounted']" :key="k" class="coverage-line"><span>{{ {direct:'Direct',comparable:'Comparable',accounted:'Accounted'}[k] }}</span>{{ fmt(r.coverage[k].count_pct, 1) }} / {{ fmt(r.coverage[k].time_pct, 1) }}</div></td><td class="conclusion">{{ r.conclusion || '—' }}</td></tr><tr v-if="!rows.length"><td colspan="7" class="empty-row">没有符合筛选条件的类型</td></tr></tbody></table></div><div class="pagination"><span>共 {{ total }} 行 · 第 {{ page + 1 }} 页</span><button :disabled="page === 0" @click="page--; action(loadResults)">上一页</button><button :disabled="(page + 1) * 30 >= total" @click="page++; action(loadResults)">下一页</button></div></template>
            <template v-if="tab === 'distribution'"><h2>类型耗时 Top 12</h2><p class="muted">NPU 全类型时间与已分配 GPU 时间并列；覆盖率低时不是完整同类性能比较。点击图形查看类型证据。</p><Chart :option="bars" @select="clickedChart"/><div class="accounting"><div v-for="kind in ['kernel', 'memory']" :key="kind"><h3>GPU {{ kind === 'kernel' ? '计算 kernel' : '内存活动' }}</h3><p>已映射 {{ summary.gpu_accounting[kind].mapped_count }} 个 / {{ fmt(summary.gpu_accounting[kind].mapped_time_us) }} μs</p><p>未映射 {{ summary.gpu_accounting[kind].unmapped_count }} 个 / {{ fmt(summary.gpu_accounting[kind].unmapped_time_us) }} μs</p></div></div><h3>GPU 未映射事件</h3></template>
            <template v-if="tab === 'coverage'"><div class="panel-heading evidence-heading"><h2>{{ scope ? `${scope.op_type} · ${scope.device_id}/${scope.core_type}` : '所有低覆盖率类型' }} / 证据</h2><button v-if="scope" @click="scope = null; recordPage = 0; action(loadRecords)">清除类型选择</button></div><div class="toolbar"><select v-model="recordKind" aria-label="证据类型"><option value="reasons">原因汇总与排查建议</option><option value="mappings">逐 kernel 映射</option><option value="diagnoses">逐 kernel 诊断</option></select><span class="muted">Accounted 不等于 Comparable；原因是候选解释，不自动证明后端缺陷。</span></div></template>
            <template v-if="tab === 'calls'"><h2>整体算子调用</h2><p class="notice">FlashAttention、CPU bundle 是重叠的替代比较视图，不能加入 TOTAL。Div/Sub 等专项证据仍在通用诊断报告中。</p></template>
            <template v-if="tab !== 'overview'"><div class="record-list"><article v-for="(r, i) in records" :key="i" class="record"><template v-if="tab === 'coverage' && recordKind === 'reasons'"><div class="record-heading"><strong>{{ r.op_type }} · {{ r.reason }}</strong><span>{{ r.kernel_count }} 个 · {{ fmt(r.npu_time_us) }} μs</span></div><p>{{ r.hypothesis }}</p><p class="small muted">CPU {{ r.cpu_op || '未知' }} · API {{ r.npu_api || '未知' }} · NPU 样本 {{ r.sample_kernel_indices?.join(', ') }}</p><ol><li v-for="step in r.next_steps" :key="step">{{ step }}</li></ol></template><template v-else><button class="type-link" @click="showRecord(r)">{{ r['Fused OP'] || r['CPU Bundle'] || r['NPU Kernel Name'] || r['GPU Name'] || r.op_type || '事件' }} <span v-if="r['NPU Kernel Index'] !== undefined">#{{ r['NPU Kernel Index'] }}</span></button><p class="muted small">{{ r.source || r['Mapping Mode'] || r.reason || r['Status'] || '查看结构化证据' }}</p><button @click="showRecord(r)">查看详情 →</button></template></article><p v-if="!records.length" class="muted">没有对应记录。无诊断可能是类型未命中阈值，不代表已证明等价。</p></div><div class="pagination"><span>共 {{ recordTotal }} 条 · 第 {{ recordPage + 1 }} 页</span><button :disabled="recordPage === 0" @click="recordPage--; action(loadRecords)">上一页</button><button :disabled="(recordPage + 1) * 20 >= recordTotal" @click="recordPage++; action(loadRecords)">下一页</button></div></template>
          </section>
        </template>
      </template>
      <footer class="main-footer">PROFILING WORKBENCH <span>证据优先 · 独立计时 · 可追溯结果</span></footer>
    </main>
    <div v-if="detail" class="drawer-backdrop" @click.self="detail = null"><section class="drawer" role="dialog" aria-label="事件证据"><button class="close" @click="detail = null" aria-label="关闭详情">×</button><h2>事件与关联证据</h2><p class="muted">编号只在本次运行内有效。原始 CSV 字段保留字符串类型。</p><pre>{{ JSON.stringify(detail, null, 2) }}</pre><template v-if="evidence"><h3>诊断证据</h3><pre>{{ JSON.stringify(evidence.items, null, 2) }}</pre><p v-if="!evidence.total">当前阈值下无逐 kernel 诊断。</p></template></section></div>
  </div>
</template>
