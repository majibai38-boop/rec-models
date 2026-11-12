<script setup>
import { computed, ref, onMounted, onBeforeUnmount, nextTick, watch } from 'vue'
const props = defineProps({job:Object, scope:Object, api:Function})
const emit = defineEmits(['scope','associate'])
const dialog=ref(null), config=ref(null), provider=ref(''), question=ref('分析当前范围的性能与覆盖率缺口，引用具体证据，并推荐值得人工核对的 GPU 关联候选。')
const mode=ref('auto'), exclusions=ref(''), followup=ref(false)
const traceMode=ref('none'), traceInfo=ref(null), traceError=ref('')
const canReuseTrace=computed(()=>followup.value&&!stale.value&&run.value?.trace_context?.reusable&&provider.value===run.value?.provider)
const traceBlocked=computed(()=>traceMode.value==='attach'? !traceInfo.value?.within_limit : traceMode.value==='reuse'&&!canReuseTrace.value)
const bytes=n=>n>=1048576?`${(n/1048576).toFixed(2)} MiB`:`${((n||0)/1024).toFixed(1)} KiB`
watch([provider,followup],()=>{if(traceMode.value==='reuse'&&!canReuseTrace.value)traceMode.value='none'})
const templates=[['overall','整体性能诊断','分析整体性能差距、主要热点和优化优先级，区分可比与未可比部分。'],['coverage','覆盖率排查','分析不可比耗时最大的类型，给出缺失证据和排查顺序。'],['shape','Shape / dtype','分析 shape 和 dtype 与性能差异的关系，列出具体样本和局限。'],['mapping','关联建议','核对 CPU 调用结构和工作量，推荐值得人工确认的关联候选。']]
function newConversation(){followup.value=false;run.value=null;traceMode.value='none';consent.value=false;clearTimeout(timer);sequence++;question.value=''}
const consent=ref(false), loading=ref(false), error=ref(''), run=ref(null), history=ref([]), deleting=ref(false), evidenceId=ref('')
const active=computed(()=>['queued','running'].includes(run.value?.status))
const totals=computed(()=>run.value?.evidence?.items.find(e=>e.kind==='scope_totals')?.data)
const stale=computed(()=>run.value?.stale || run.value?.attempt!==props.job.attempt || props.job.status!=='succeeded')
const selectedProvider=computed(()=>config.value?.providers.find(p=>p.id===provider.value))
const status={queued:'等待 AI 分析',running:'正在分析证据',succeeded:'分析完成',failed:'分析失败',cancelled:'已取消',interrupted:'已中断'}
const confidence={high:'建议优先核对',medium:'仍需补充证据',low:'证据较弱'}
let timer, disposed=false, sequence=0, previousFocus
const url=()=>`/jobs/${props.job.id}/ai-runs`
async function action(fn) {loading.value=true;error.value='';try{await fn()}catch(e){error.value=e.message}finally{loading.value=false}}
async function refreshHistory(){history.value=await props.api(url())}
async function select(id) {
  const seq=++sequence;clearTimeout(timer);deleting.value=false;evidenceId.value=''
  const next=await props.api(`${url()}/${id}`)
  if(disposed || seq!==sequence)return
  run.value=next;followup.value=next.status==='succeeded'&&!next.stale
  provider.value=next.provider
  traceMode.value=next.trace_context?.reusable&&!next.stale?'reuse':'none'
  if(['queued','running'].includes(next.status))timer=setTimeout(()=>poll(id,seq),1500)
}
async function poll(id,seq){
  try{const next=await props.api(`${url()}/${id}`);if(disposed||seq!==sequence)return;run.value=next;followup.value=next.status==='succeeded'&&!next.stale
    if(['queued','running'].includes(next.status))timer=setTimeout(()=>poll(id,seq),1500)
    else {if(canReuseTrace.value)traceMode.value='reuse';else traceMode.value='none';await refreshHistory()}
  }catch(e){if(!disposed&&seq===sequence)error.value=e.message}
}
async function start(){await action(async()=>{
  const created=await props.api(url(),'POST',{attempt:props.job.attempt,provider:provider.value,question:question.value.trim(),
    parent_id:followup.value&&!stale.value?run.value?.id:null,mode:mode.value,trace_mode:traceMode.value,exclude_types:exclusions.value.split(/[,，]/).map(s=>s.trim()).filter(Boolean),
    scope:props.scope?.op_type?{op_type:props.scope.op_type,device_id:props.scope.device_id,core_type:props.scope.core_type}:null,share_evidence:consent.value})
  await refreshHistory();await select(created.id)
})}
async function cancel(){await action(async()=>{await props.api(`${url()}/${run.value.id}/cancel`,'POST');await select(run.value.id);await refreshHistory()})}
async function remove(){await action(async()=>{await props.api(`${url()}/${run.value.id}`,'DELETE');run.value=null;traceMode.value='none';followup.value=false;deleting.value=false;await refreshHistory()})}
function associate(proposal){
  const row=run.value.evidence.items.find(e=>e.kind==='type'&&['op_type','device_id','core_type'].every(k=>e.data[k]===proposal.type[k]))?.data
  if(!row||stale.value)return
  emit('associate',{type:row,seed:{...proposal,attempt:run.value.attempt,ai_run_id:run.value.id,ai_proposal_id:proposal.id}})
}
async function showEvidence(id){evidenceId.value=id;await nextTick();dialog.value.querySelector('[data-ai-evidence]')?.scrollIntoView({block:'nearest'})}
function download(){const blob=new Blob([JSON.stringify(run.value,null,2)],{type:'application/json'});const link=document.createElement('a');const url=URL.createObjectURL(blob);link.href=url;link.download='ai-analysis-'+run.value.id+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
onMounted(async()=>{previousFocus=document.activeElement;dialog.value.focus();await action(async()=>{
  try{traceInfo.value=await props.api(`/jobs/${props.job.id}/ai-trace?attempt=${props.job.attempt}`)}catch(e){traceError.value=e.message}
  const [cfg,runs]=await Promise.all([props.api('/ai/config'),props.api(url())]);if(disposed)return
  config.value=cfg;history.value=runs;provider.value=cfg.providers.find(p=>p.available)?.id||cfg.providers[0]?.id||''
  if(runs.length)await select(runs[0].id)
})})
onBeforeUnmount(()=>{disposed=true;sequence++;clearTimeout(timer);previousFocus?.focus()})
</script>

<template>
  <div class="ai-workspace">
    <section ref="dialog" class="ai-dialog" aria-labelledby="ai-title" tabindex="-1">
      <div class="toolbar"><button :disabled="active" @click="newConversation">新建会话</button><button v-if="scope?.op_type" :disabled="active" @click="emit('scope',{})">切换到全任务</button></div>
      <h2 id="ai-title">AI 辅助分析</h2>
      <p class="muted small">GPU Step {{ job.options?.all_steps?'全量':job.options?.gpu_step }} · NPU Step {{ job.options?.all_steps?'全量':job.options?.npu_step }} · 同一结果版本内追问，版本变化后需新建会话。</p>
      <p class="muted">{{ scope?.op_type ? `${scope.op_type} / ${scope.device_id} / ${scope.core_type}` : '全任务 · 完整统计 + 按问题选择明细' }} · 结果 {{ job.attempt.slice(0,8) }}</p>
      <p class="notice">AI 提供证据解释和关联建议。关联需人工预览确认，可在「人工补充关联」中撤销。切换页签后分析继续，历史可恢复查看。</p>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
      <div class="toolbar"><button v-for="t in templates" :key="t[0]" :disabled="active" @click="mode=t[0];question=t[2]">{{ t[1] }}</button></div>
      <template v-if="config">
        <label>分析后端<select v-model="provider" :disabled="loading||active"><option v-for="p in config.providers" :key="p.id" :value="p.id">{{ p.label }}{{ p.available?'':' · 未就绪' }}</option></select></label>
        <p v-if="!selectedProvider?.available" class="notice">{{ selectedProvider?.reason }}。配置方式见部署文档「AI 辅助分析」。</p>
        <p class="muted small">{{ selectedProvider?.model }} · 超时 {{ config.timeout_seconds }} 秒 · CLI 在服务器运行，客户端无需安装。</p>
        <label>证据查询方式<select v-model="mode" :disabled="active"><option value="auto">按问题自动选择</option><option v-for="t in templates" :key="t[0]" :value="t[0]">{{ t[1] }}</option></select></label><label>排除的 NPU 类型（逗号分隔，精确名称）<input v-model="exclusions" :disabled="active" placeholder="例如 Cast, Slice" /></label><p class="muted small">完整汇总不受明细采样影响。排除后的差额是累计 kernel 时间变化，不是端到端收益。</p>
        <label v-if="run?.status==='succeeded'&&!stale" class="check"><input type="checkbox" v-model="followup">继续当前会话（继承最近 4 轮，可调整分析范围）</label>
        <label>完整 Trace<select v-model="traceMode" :disabled="loading||active">
          <option value="none">不发送完整 Trace（仅统计证据）</option>
          <option value="attach" :disabled="!traceInfo?.within_limit">{{ run?.evidence?.trace?'重新附加':'首次附加' }}本次选中 Step 的完整 GPU/NPU Trace</option>
          <option value="reuse" :disabled="!canReuseTrace">复用已发送的 Trace（不重复附加文件）</option>
        </select></label>
        <p v-if="traceInfo" class="muted small">{{ traceInfo.files.map(f=>`${f.name}：${bytes(f.bytes)}`).join(' · ') }} · 合计 {{ bytes(traceInfo.total_bytes) }} · 发送上限 {{ bytes(traceInfo.max_bytes) }}</p>
        <p v-if="traceInfo&&!traceInfo.within_limit" class="notice">完整 Trace 超过发送上限，不会截断发送。可缩小 Step 范围或仅使用统计证据；管理员可根据模型上下文容量调整上限。</p>
        <p v-if="traceError" class="notice">完整 Trace 暂不可用：{{ traceError }}。仍可仅使用统计证据。</p>
        <p v-if="traceMode==='attach'" class="notice">将发送本次所选 Step/PID 范围的全部 Trace 内容，包含原始事件参数、路径等元数据；类型筛选与排除类型不会裁剪该文件。发送后自动启用会话复用。重新附加会创建独立后端会话。</p>
        <p v-if="traceMode==='reuse'" class="notice">本轮不重复附加 JSON，继续同一后端会话。模型服务仍可能传输、计费或压缩历史上下文，不能保证零重复 Token。</p>
        <p v-if="run?.evidence?.trace&&!run?.trace_context?.reusable" class="muted small">{{ run.trace_context?.reason }}。重新附加需主动选择，不会自动重发。</p>
        <p v-if="traceMode==='none'&&run?.evidence?.trace" class="muted small">本轮不继续含 Trace 的后端会话，仅发送统计证据和勾选追问时的最近 4 轮问答摘要；此前已发送的数据无法撤回。</p>
        <label>想分析的问题<textarea v-model="question" :disabled="loading||active" maxlength="2000" rows="3"/></label>
        <label class="check"><input type="checkbox" v-model="consent" :disabled="loading||active">允许按上述选择，将问题、统计证据{{ traceMode==='attach'?'及完整 Trace':traceMode==='reuse'?'及已发送的 Trace 会话上下文':'' }}发送给所选模型服务。</label>
        <button class="primary" :disabled="loading||active||!consent||traceBlocked||!question.trim()||!selectedProvider?.available||job.status!=='succeeded'" @click="start">{{ followup&&!stale?'发送追问':'开始 AI 分析' }}</button>
      </template>
      <div v-if="history.length" class="ai-history"><label>分析历史<select :value="run?.id||''" :disabled="loading" @change="action(()=>select($event.target.value))"><option value="" disabled>选择分析</option><option v-for="r in history" :key="r.id" :value="r.id">{{ new Date(r.created*1000).toLocaleString() }} · {{ r.provider }} · {{ status[r.status] }}</option></select></label></div>
      <section v-if="run" class="ai-result" aria-live="polite">
        <div v-if="totals" class="metrics"><article><span>范围内 NPU 累计时间</span><strong>{{ totals.before.npu_time_us }} μs</strong></article><article><span>不可比 NPU 累计时间</span><strong>{{ totals.before.uncomparable_npu_time_us }} μs</strong></article><article><span>排除后 NPU 累计时间</span><strong>{{ totals.after_exclusions.npu_time_us }} μs</strong></article></div>
        <div class="toolbar"><strong>{{ status[run.status] }} · {{ run.provider }}</strong><button v-if="active" :disabled="loading" @click="cancel">取消分析</button><button v-if="!active" @click="download">下载分析 JSON</button><button v-if="!active" @click="deleting=!deleting">删除此记录</button><button v-if="active&&error" @click="action(()=>select(run.id))">刷新状态</button></div>
        <div v-if="deleting" class="notice">删除这次 AI 分析记录，已确认的人工关联保留。删除会话最新记录后不能从旧轮次继续复用；最后一条引用删除时清理本地 Trace 会话。<button :disabled="loading" @click="remove">确认删除记录</button></div>
        <p v-if="stale" class="notice">此分析属于旧结果或任务正在更新，可阅读历史，但不能应用其中的建议。</p>
        <p v-if="run.error" class="error">{{ run.error }}</p>
        <p v-if="run.evidence?.trace" class="notice">本轮 Trace：{{ run.evidence.trace.mode==='reuse'?'复用后端会话':'附加完整文件' }} · 本轮新增附件 {{ bytes(run.evidence.trace.attached_bytes) }} · {{ run.trace_context?.reusable?'可继续复用':'尚不可复用' }}</p>
        <details><summary>本次问题与范围</summary><p>{{ run.request.question }}</p><pre>{{ run.request.scope || '全任务' }}</pre></details>
        <details v-if="run.evidence?.conversation?.length"><summary>本轮继承的对话 · {{ run.evidence.conversation.length }} 轮</summary><article v-for="(turn,i) in run.evidence.conversation" :key="i" class="ai-card"><strong>{{ turn.question }}</strong><p>{{ turn.answer }}</p></article></details>
        <template v-if="run.result">
          <p class="ai-summary">{{ run.result.summary }}</p>
          <article v-for="(f,i) in run.result.findings" :key="i" class="ai-card"><h3>{{ f.title }}</h3><p>{{ f.detail }}</p><button v-for="id in f.evidence_ids" :key="id" class="type-link" @click="showEvidence(id)">{{ id }} ↗</button></article>
          <h3>建议核对的关联 · {{ run.result.proposals.length }}</h3>
          <p v-if="!run.result.proposals.length" class="muted">本次没有足够证据推荐关联。可缩小到具体类型，或人工检查更多 GPU 候选。</p>
          <article v-for="p in run.result.proposals" :key="p.id" class="ai-card"><strong>{{ p.type.op_type }} · {{ confidence[p.confidence] }}</strong><p>{{ p.target_id }} → {{ p.candidate_id }}</p><p>{{ p.reason }}</p><div class="toolbar"><button v-for="id in p.evidence_ids" :key="id" class="type-link" @click="showEvidence(id)">{{ id }} ↗</button><button :disabled="stale||loading" @click="associate(p)">核对并预览关联 →</button></div></article>
          <h3 v-if="run.result.limitations.length">分析局限</h3><ul><li v-for="(text,i) in run.result.limitations" :key="i">{{ text }}</li></ul>
        </template>
        <details v-if="run.evidence" :open="!!evidenceId" data-ai-evidence><summary>模型收到的证据 · {{ run.evidence.items.length }} 条</summary><p class="muted small">{{ run.evidence.limits.sampling }}</p><p v-if="run.evidence.limits.omitted.length">超出大小上限，已省略 {{ run.evidence.limits.omitted.length }} 条。</p><label>证据<select v-model="evidenceId"><option value="">选择证据</option><option v-for="e in run.evidence.items" :key="e.id" :value="e.id">{{ e.id }} · {{ e.kind }}</option></select></label><pre v-if="evidenceId">{{ JSON.stringify(run.evidence.items.find(e=>e.id===evidenceId)?.data,null,2) }}</pre></details>
      </section>
    </section>
  </div>
</template>

<style scoped>
.ai-dialog{width:100%;outline:none}.ai-workspace{max-width:1200px;margin:auto}.ai-result{display:grid;grid-template-columns:minmax(0,1fr);gap:10px}.ai-workspace input:not([type=checkbox]){width:100%;padding:8px}.ai-workspace .toolbar{flex-wrap:wrap}
select,textarea{width:100%;margin:6px 0 12px}
.ai-dialog p,.ai-dialog li{white-space:pre-wrap;overflow-wrap:anywhere}
.ai-dialog pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto;background:#f4f7f8;padding:12px}
.ai-card{border:1px solid #d8e1e5;border-radius:8px;padding:14px;margin:12px 0}
.ai-card h3{margin-top:0}.ai-card .type-link{margin-right:12px}
.ai-history,.ai-result{border-top:1px solid #d8e1e5;margin-top:20px;padding-top:16px}
.ai-summary{font-size:1.05rem;line-height:1.7}.ai-dialog .check{margin:12px 0}
</style>
