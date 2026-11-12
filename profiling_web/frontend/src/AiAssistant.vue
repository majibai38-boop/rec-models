<script setup>
import { computed, ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
const props = defineProps({job:Object, scope:Object, api:Function})
const emit = defineEmits(['close','associate'])
const dialog=ref(null), config=ref(null), provider=ref(''), question=ref('分析当前范围的性能与覆盖率缺口，引用具体证据，并推荐值得人工核对的 GPU 关联候选。')
const consent=ref(false), loading=ref(false), error=ref(''), run=ref(null), history=ref([]), deleting=ref(false), evidenceId=ref('')
const active=computed(()=>['queued','running'].includes(run.value?.status))
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
  run.value=next
  if(['queued','running'].includes(next.status))timer=setTimeout(()=>poll(id,seq),1500)
}
async function poll(id,seq){
  try{const next=await props.api(`${url()}/${id}`);if(disposed||seq!==sequence)return;run.value=next
    if(['queued','running'].includes(next.status))timer=setTimeout(()=>poll(id,seq),1500)
    else await refreshHistory()
  }catch(e){if(!disposed&&seq===sequence)error.value=e.message}
}
async function start(){await action(async()=>{
  const created=await props.api(url(),'POST',{attempt:props.job.attempt,provider:provider.value,question:question.value.trim(),
    scope:props.scope?.op_type?{op_type:props.scope.op_type,device_id:props.scope.device_id,core_type:props.scope.core_type}:null,share_evidence:consent.value})
  await refreshHistory();await select(created.id)
})}
async function cancel(){await action(async()=>{await props.api(`${url()}/${run.value.id}/cancel`,'POST');await select(run.value.id);await refreshHistory()})}
async function remove(){await action(async()=>{await props.api(`${url()}/${run.value.id}`,'DELETE');run.value=null;deleting.value=false;await refreshHistory()})}
function associate(proposal){
  const row=run.value.evidence.items.find(e=>e.kind==='type'&&['op_type','device_id','core_type'].every(k=>e.data[k]===proposal.type[k]))?.data
  if(!row||stale.value)return
  emit('associate',{type:row,seed:{...proposal,attempt:run.value.attempt,ai_run_id:run.value.id,ai_proposal_id:proposal.id}})
}
async function showEvidence(id){evidenceId.value=id;await nextTick();dialog.value.querySelector('[data-ai-evidence]')?.scrollIntoView({block:'nearest'})}
function download(){const blob=new Blob([JSON.stringify(run.value,null,2)],{type:'application/json'});const link=document.createElement('a');const url=URL.createObjectURL(blob);link.href=url;link.download='ai-analysis-'+run.value.id+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function key(e){
  if(e.key==='Escape'){e.preventDefault();emit('close')}
  if(e.key!=='Tab')return
  const elements=[...dialog.value.querySelectorAll('button,input,select,textarea,summary')].filter(x=>!x.disabled&&x.getClientRects().length)
  if(e.shiftKey&&document.activeElement===elements[0]){e.preventDefault();elements.at(-1)?.focus()}
  else if(!e.shiftKey&&document.activeElement===elements.at(-1)){e.preventDefault();elements[0]?.focus()}
}
onMounted(async()=>{previousFocus=document.activeElement;dialog.value.focus();document.addEventListener('keydown',key);await action(async()=>{
  const [cfg,runs]=await Promise.all([props.api('/ai/config'),props.api(url())]);if(disposed)return
  config.value=cfg;history.value=runs;provider.value=cfg.providers.find(p=>p.available)?.id||cfg.providers[0]?.id||''
  if(runs.length)await select(runs[0].id)
})})
onBeforeUnmount(()=>{disposed=true;sequence++;clearTimeout(timer);document.removeEventListener('keydown',key);previousFocus?.focus()})
</script>

<template>
  <div class="drawer-backdrop" @click.self="emit('close')">
    <section ref="dialog" class="drawer ai-dialog" role="dialog" aria-modal="true" aria-labelledby="ai-title" tabindex="-1">
      <button class="close" aria-label="关闭 AI 辅助" @click="emit('close')">×</button>
      <h2 id="ai-title">AI 辅助分析</h2>
      <p class="muted">{{ scope?.op_type ? `${scope.op_type} / ${scope.device_id} / ${scope.core_type}` : '全任务 · 耗时最高的前 6 个类型' }} · 结果 {{ job.attempt.slice(0,8) }}</p>
      <p class="notice">AI 提供证据解释和关联建议。关联需人工预览确认，可在「人工补充关联」中撤销。关闭窗口后分析继续。</p>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
      <template v-if="config">
        <label>分析后端<select v-model="provider" :disabled="loading||active"><option v-for="p in config.providers" :key="p.id" :value="p.id">{{ p.label }}{{ p.available?'':' · 未就绪' }}</option></select></label>
        <p v-if="!selectedProvider?.available" class="notice">{{ selectedProvider?.reason }}。配置方式见部署文档「AI 辅助分析」。</p>
        <p class="muted small">{{ selectedProvider?.model }} · 超时 {{ config.timeout_seconds }} 秒 · CLI 在服务器运行，客户端无需安装。</p>
        <label>想分析的问题<textarea v-model="question" :disabled="loading||active" maxlength="2000" rows="3"/></label>
        <label class="check"><input type="checkbox" v-model="consent" :disabled="loading||active">允许将问题及当前范围的统计、算子名称、模块、shape/dtype、调用样本发送给所选模型服务。不发送完整 Trace。</label>
        <button class="primary" :disabled="loading||active||!consent||!question.trim()||!selectedProvider?.available||job.status!=='succeeded'" @click="start">开始 AI 分析</button>
      </template>
      <div v-if="history.length" class="ai-history"><label>分析历史<select :value="run?.id||''" :disabled="loading" @change="action(()=>select($event.target.value))"><option value="" disabled>选择分析</option><option v-for="r in history" :key="r.id" :value="r.id">{{ new Date(r.created*1000).toLocaleString() }} · {{ r.provider }} · {{ status[r.status] }}</option></select></label></div>
      <section v-if="run" class="ai-result" aria-live="polite">
        <div class="toolbar"><strong>{{ status[run.status] }} · {{ run.provider }}</strong><button v-if="active" :disabled="loading" @click="cancel">取消分析</button><button v-if="!active" @click="download">下载分析 JSON</button><button v-if="!active" @click="deleting=!deleting">删除此记录</button><button v-if="active&&error" @click="action(()=>select(run.id))">刷新状态</button></div>
        <div v-if="deleting" class="notice">仅删除这次 AI 分析记录，已确认的人工关联保留。<button :disabled="loading" @click="remove">确认删除记录</button></div>
        <p v-if="stale" class="notice">此分析属于旧结果或任务正在更新，可阅读历史，但不能应用其中的建议。</p>
        <p v-if="run.error" class="error">{{ run.error }}</p>
        <details><summary>本次问题与范围</summary><p>{{ run.request.question }}</p><pre>{{ run.request.scope || '全任务 Top 6 类型' }}</pre></details>
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
.ai-dialog{width:min(900px,96vw)}
select,textarea{width:100%;margin:6px 0 12px}
.ai-dialog p,.ai-dialog li{white-space:pre-wrap;overflow-wrap:anywhere}
.ai-dialog pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto;background:#f4f7f8;padding:12px}
.ai-card{border:1px solid #d8e1e5;border-radius:8px;padding:14px;margin:12px 0}
.ai-card h3{margin-top:0}.ai-card .type-link{margin-right:12px}
.ai-history,.ai-result{border-top:1px solid #d8e1e5;margin-top:20px;padding-top:16px}
.ai-summary{font-size:1.05rem;line-height:1.7}.ai-dialog .check{margin:12px 0}
</style>
