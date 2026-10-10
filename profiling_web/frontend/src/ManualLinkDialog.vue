<script setup>
import { computed, ref, watch, onMounted, onBeforeUnmount } from 'vue'
const props = defineProps({ job: Object, type: Object, api: Function })
const emit = defineEmits(['close', 'submitted', 'upgrade'])
const data = ref(null), selected = ref(''), targetId = ref(''), search = ref(''), reason = ref('')
const loading = ref(false), saving = ref(false), error = ref(''), offset = ref(0), gpuOffset = ref(0), dialog = ref(null)
const pendingUndo = ref('')
const batchPreview = ref(null), preparingBatch = ref(false)
let batchSequence = 0
let sequence = 0, previousFocus
const fmt = (n) => Number(n || 0).toLocaleString('en-US', { maximumFractionDigits: 3 })
const candidate = computed(() => data.value?.candidates.find(r => r.id === selected.value))
const target = computed(() => data.value?.target)
const batchCoverage = computed(() => props.type.npu_total_time_us > 0
  ? fmt(Math.min(100, 100 * ((props.type.npu_comparable_time_us || 0) + (batchPreview.value?.npu_time_us || 0)) / props.type.npu_total_time_us)) + '%'
  : '—')
watch(selected, () => {batchSequence++; batchPreview.value=null; preparingBatch.value=false})
const coverage = computed(() => props.type.npu_total_time_us > 0
  ? fmt(Math.min(100, 100 * ((props.type.npu_comparable_time_us || 0) + (target.value?.time_us || 0)) / props.type.npu_total_time_us)) + '%'
  : '—')
async function load(resetTarget = false) {
  const seq = ++sequence
  loading.value = true; error.value = ''; selected.value = ''
  if (resetTarget) targetId.value = ''
  try {
    const params = {op_type:props.type.op_type, device_id:props.type.device_id, core_type:props.type.core_type,
      target_id:targetId.value, search:search.value, offset:offset.value, gpu_offset:gpuOffset.value}
    const result = await props.api(`/jobs/${props.job.id}/manual-options?` + new URLSearchParams(params))
    if (seq !== sequence) return
    if (result.attempt !== props.job.attempt) throw Error('结果已更新，请关闭弹窗并刷新任务后重试')
    data.value = result; targetId.value = result.target?.id || ''
  } catch (e) {if (seq === sequence) error.value = e.message}
  finally {if (seq === sequence) loading.value = false}
}
async function submit() {
  if (!candidate.value || !reason.value.trim() || saving.value || loading.value) return
  saving.value = true; error.value = ''
  try {
    const job = await props.api(`/jobs/${props.job.id}/manual-links`, 'POST', {
      attempt:data.value.attempt, target_id:targetId.value, candidate_id:selected.value, reason:reason.value.trim()})
    emit('submitted', job)
  } catch(e) {error.value = e.message} finally {saving.value = false}
}
async function previewBatch() {
  if (!candidate.value || loading.value || saving.value) return
  const seq = ++batchSequence
  preparingBatch.value=true; batchPreview.value=null; error.value=''
  try {
    const result = await props.api(`/jobs/${props.job.id}/manual-batch/preview`, 'POST', {
      attempt:data.value.attempt, target_id:targetId.value, candidate_id:selected.value})
    if (seq === batchSequence) batchPreview.value=result
  } catch(e) {if (seq === batchSequence) error.value=e.message}
  finally {if (seq === batchSequence) preparingBatch.value=false}
}
async function submitBatch() {
  if (!batchPreview.value || !reason.value.trim() || saving.value || loading.value) return
  saving.value=true; error.value=''
  try {
    emit('submitted', await props.api(`/jobs/${props.job.id}/manual-batch`, 'POST', {
      attempt:data.value.attempt, target_id:targetId.value, candidate_id:selected.value,
      preview_token:batchPreview.value.preview_token, reason:reason.value.trim()}))
  } catch(e) {error.value=e.message} finally {saving.value=false}
}
async function undo(link) {
  saving.value = true; error.value = ''
  try {emit('submitted', await props.api(`/jobs/${props.job.id}/manual-links/${link.id}/undo`, 'POST', {attempt:data.value.attempt}))}
  catch(e) {error.value = e.message} finally {saving.value = false}
}
function key(e) {
  if (e.key === 'Escape' && !saving.value) emit('close')
  if (e.key !== 'Tab') return
  const elements = [...dialog.value.querySelectorAll('button,input,select,textarea,summary')].filter(x => !x.disabled && x.getClientRects().length)
  const first = elements[0], last = elements.at(-1)
  if (e.shiftKey && document.activeElement === first) {e.preventDefault(); last?.focus()}
  else if (!e.shiftKey && document.activeElement === last) {e.preventDefault(); first?.focus()}
}
onMounted(() => {previousFocus = document.activeElement; dialog.value.focus(); document.addEventListener('keydown',key); load()})
onBeforeUnmount(() => {sequence++; batchSequence++; document.removeEventListener('keydown',key); previousFocus?.focus()})
</script>

<template>
  <div class="drawer-backdrop" @click.self="!saving && $emit('close')">
    <section ref="dialog" class="drawer manual-dialog" role="dialog" aria-modal="true" aria-labelledby="manual-title" tabindex="-1" :aria-busy="loading || saving">
      <button class="close" aria-label="关闭补充关联" :disabled="saving" @click="$emit('close')">×</button>
      <h2 id="manual-title">{{ type.op_type }} · 补充关联</h2>
      <p class="muted">{{ type.device_id }} / {{ type.core_type }} · 当前可比时间覆盖率 {{ fmt(type.coverage.comparable.time_pct) }}%</p>
      <p class="notice">只关联尚未分配的 GPU 计算 kernel，按整组比较；不修改真实 CPU owner，不逐个分摊时间。人工确认不等于自动验证。</p>
      <p v-if="error" role="alert" class="error">{{ error }}</p>
      <p v-if="loading" role="status">正在查找候选…</p>
      <template v-if="data && !loading">
        <p v-if="!data.supported" class="notice">这份旧报告缺少候选索引。请复用输入并重新分析，原报告不会被覆盖。<button @click="$emit('upgrade')">复用输入</button></p>
        <template v-else>
          <h3>1. 选择未可比的 NPU 调用</h3>
          <select v-if="data.targets.length" v-model="targetId" :disabled="saving" aria-label="NPU 调用" @change="gpuOffset=0;load()">
            <option v-for="r in data.targets" :key="r.id" :value="r.id">{{ r.cpu_name }} · CPU #{{ r.cpu_index ?? '未知' }} · {{ r.count }} kernels · {{ fmt(r.time_us) }} μs</option>
          </select>
          <p v-else>该类型没有可补充的 NPU kernel。</p>
          <div v-if="data.target_total > 20" class="pagination"><span>共 {{ data.target_total }} 组</span><button :disabled="saving || offset===0" @click="offset-=20;gpuOffset=0;load(true)">上一页 NPU</button><button :disabled="saving || offset+20>=data.target_total" @click="offset+=20;gpuOffset=0;load(true)">下一页 NPU</button></div>
          <div v-if="target" class="manual-target">
            <p>NPU #{{ target.indices.join(', ') }} · 模块 {{ target.module || '未知' }} · {{ fmt(target.time_us) }} μs</p>
            <details><summary>NPU kernel 与元数据</summary><pre>{{ target.names.join('\n') }}
Shape: {{ JSON.stringify(target.shapes) }}
Dtype: {{ JSON.stringify(target.dtypes) }}</pre></details>
            <h3>2. 选择 GPU 候选</h3>
            <form class="toolbar" @submit.prevent="gpuOffset=0;load()"><input v-model="search" maxlength="200" placeholder="搜索其他未映射 GPU kernel / CPU op / 模块" aria-label="搜索 GPU 候选" :disabled="saving"><button :disabled="saving">搜索</button><button type="button" :disabled="saving" @click="search='';gpuOffset=0;load()">前三个推荐</button></form>
            <p class="muted small">{{ search ? '搜索结果' : '证据最充分的前三个候选' }}；推荐排序不是匹配概率，耗时接近不作为依据。</p>
            <article v-for="r in data.candidates" :key="r.id" class="manual-candidate" :class="{chosen:selected===r.id}">
              <label><input v-model="selected" type="radio" name="gpu-candidate" :value="r.id" :disabled="saving"> <strong>{{ r.recommendation }} · {{ r.cpu_name }}</strong></label>
              <p>GPU #{{ r.indices.join(', ') }} · {{ r.count }} kernels · {{ fmt(r.time_us) }} μs · {{ r.module || '模块未知' }}</p>
              <p>{{ r.reasons.join('；') }}</p><p v-if="r.warnings.length" class="notice">{{ r.warnings.join('；') }}，请人工核实。</p>
              <details><summary>kernel 名称与元数据</summary><pre>{{ r.names.join('\n') }}
Shape: {{ JSON.stringify(r.shapes) }}
Dtype: {{ JSON.stringify(r.dtypes) }}</pre></details>
            </article>
            <p v-if="!data.candidates.length">没有可用候选。已分配、跨 GPU 设备或含 GPU 内存活动的调用不在本版候选中。</p>
            <div v-if="search && data.candidate_total > data.candidate_limit" class="pagination"><span>共 {{ data.candidate_total }} 组</span><button :disabled="saving || gpuOffset===0" @click="gpuOffset-=20;load()">上一页 GPU</button><button :disabled="saving || gpuOffset+20>=data.candidate_total" @click="gpuOffset+=20;load()">下一页 GPU</button></div>
            <template v-if="candidate">
              <h3>3. 确认关联</h3><p>新增可比 NPU 时间 {{ fmt(target.time_us) }} μs；分配 GPU 时间 {{ fmt(candidate.time_us) }} μs。预计类型覆盖率 {{ coverage }}，所选整组 GPU/NPU 比值 {{ target.time_us ? fmt(candidate.time_us/target.time_us) : '—' }}。TOTAL 不变。</p>
              <label>确认理由<textarea v-model="reason" maxlength="500" placeholder="例如：已核对两侧 index_add_ 调用，NPU 仅多一层内部调用" :disabled="saving" /></label>
              <div class="toolbar"><button :disabled="saving || !reason.trim()" @click="submit">{{ saving ? '提交中…' : '仅确认当前组' }}</button><button class="primary" :disabled="saving || preparingBatch" @click="previewBatch">{{ preparingBatch ? '正在生成预览…' : '预览同类批量关联' }}</button></div>
              <section v-if="batchPreview" class="manual-candidate batch-preview" aria-label="批量关联预览">
                <h3>批量关联预览 · {{ batchPreview.pair_count }} 对调用</h3>
                <p>以当前选择为模板：同模块、同调用结构的 NPU {{ batchPreview.scope_target_count }} 组，GPU {{ batchPreview.scope_candidate_count }} 组。</p>
                <p>可关联 NPU {{ batchPreview.npu_kernel_count }} 个 kernel ↔ GPU {{ batchPreview.gpu_kernel_count }} 个 kernel。新增可比 NPU 时间 {{ fmt(batchPreview.npu_time_us) }} μs，分配 GPU 时间 {{ fmt(batchPreview.gpu_time_us) }} μs；预计类型覆盖率 {{ batchCoverage }}。</p>
                <p class="muted small">每对独立保存，整批只生成一次报告，可整批撤销。TOTAL 不变。</p>
                <p v-if="batchPreview.basis_counts.manual_template_order" class="notice">其中 {{ batchPreview.basis_counts.manual_template_order }} 对依据完整元数据和各端调用顺序扩展，请确认它们属于同一组重复调用。</p>
                <p v-if="batchPreview.excluded_count">同模板有 {{ batchPreview.excluded_count }} 组未纳入：<span v-for="(n,why) in batchPreview.excluded_reasons" :key="why">{{ why }}（{{ n }} 组）；</span></p>
                <p v-if="batchPreview.other_target_count" class="muted small">该类型另有 {{ batchPreview.other_target_count }} 组属于其他模块或调用结构，本次不包含。</p>
                <details><summary>查看配对样本（{{ batchPreview.pairs.length }} / {{ batchPreview.pair_count }}）</summary><div class="table-scroll"><table><thead><tr><th>NPU CPU # / kernels</th><th>GPU CPU # / kernels</th><th>依据</th></tr></thead><tbody><tr v-for="p in batchPreview.pairs" :key="p.target_id"><td>{{ p.npu_cpu_index }} / {{ p.npu_indices.length }}</td><td>{{ p.gpu_cpu_index }} / {{ p.gpu_indices.length }}</td><td>{{ p.basis==='aligned_cpu_scope' ? '已有 CPU 对齐' : '人工模板 + 调用顺序' }}</td></tr></tbody></table></div></details>
                <button class="primary" :disabled="saving || !reason.trim()" @click="submitBatch">{{ saving ? '提交中…' : `一次确认 ${batchPreview.pair_count} 对并更新报告` }}</button>
              </section>
            </template>
          </div>
          <h3 v-if="data.links.length">已人工确认（可撤销）</h3>
          <article v-for="link in data.links" :key="link.id" class="manual-candidate"><p v-if="link.pair_count>1">批量关联 {{ link.pair_count }} 对调用 · NPU {{ link.npu_indices.length }} / GPU {{ link.gpu_indices.length }} 个 kernel</p><p v-else>NPU #{{ link.npu_indices.join(', ') }} ↔ GPU #{{ link.gpu_indices.join(', ') }}</p><p>{{ link.reason }}</p><template v-if="pendingUndo===link.id"><p>撤销{{ link.pair_count>1 ? '整批 '+link.pair_count+' 对关联' : '关联' }}后重新生成报告，自动结果会保留。</p><button :disabled="saving" @click="undo(link)">确认撤销</button><button :disabled="saving" @click="pendingUndo=''">取消撤销</button></template><button v-else :disabled="saving" @click="pendingUndo=link.id">{{ link.pair_count>1 ? '撤销整批' : '撤销关联' }}</button></article>
        </template>
      </template>
    </section>
  </div>
</template>

<style scoped>
.manual-dialog { width:min(880px,96vw); }
.manual-candidate { border:1px solid #d8e1e5; padding:14px; margin:12px 0; border-radius:8px; }
.manual-candidate.chosen { border-color:#167d78; background:#eff9f7; }
.manual-dialog select,.manual-dialog textarea { width:100%; }
.manual-dialog textarea { display:block; min-height:75px; margin:8px 0 14px; }
.manual-dialog pre { white-space:pre-wrap; overflow-wrap:anywhere; }
.manual-dialog p { overflow-wrap:anywhere; }
.manual-dialog label { cursor:pointer; }
.batch-preview>.primary { margin-top:16px; }
</style>
