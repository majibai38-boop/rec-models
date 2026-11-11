<script setup>
import {computed,ref,reactive,watch,nextTick,onBeforeUnmount} from 'vue'
import {traceRange,viewerUrl,waitForViewer,sendTrace} from './traceViewer.js'
const props=defineProps({job:Object,unit:Object,attempt:String,api:Function})
const active=ref(''),expanded=ref(false),visible=ref(false),error=ref(''),eventIndex=ref(0)
const whole=ref(false),padding=ref(1),frames={},controllers={}
const sessions=reactive({npu:null,gpu:null})
let generation=0
const events=computed(()=>props.unit?.[active.value+'_kernels'] || [])
const current=computed(()=>events.value[eventIndex.value])
const range=computed(()=>traceRange(props.unit,active.value,eventIndex.value,whole.value,padding.value))
const key=computed(()=>props.job.id+':'+(props.attempt||props.job.attempt))
function release(side) {controllers[side]?.abort();delete controllers[side];sessions[side]=null;delete frames[side]}
function reset() {generation++;release('gpu');release('npu');visible.value=false;active.value='';expanded.value=false;error.value=''}
watch(key,reset)
watch(()=>props.unit?.id,()=>{
  eventIndex.value=0;whole.value=false;padding.value=1;error.value=''
  if(props.unit) {
    if(!props.unit[active.value+'_kernels']?.length){visible.value=false;expanded.value=false}
    nextTick(focus)
  }
})
onBeforeUnmount(reset)
function focus() {
  const target=frames[active.value]?.contentWindow
  if(target && range.value) target.postMessage({perfetto:range.value},location.origin)
}
async function open(side) {
  active.value=side;visible.value=true;eventIndex.value=0;whole.value=false;padding.value=1;error.value=''
  if(sessions[side]) {await nextTick();focus();return}
  controllers[side]?.abort()
  const gen=generation, controller=new AbortController();controllers[side]=controller
  try {
    const config=await props.api('/trace-viewer')
    if(gen!==generation || controller.signal.aborted)return
    if(!config.available)throw new Error('服务尚未安装 Perfetto。请在部署目录运行 python -m profiling_web.perfetto 并重启服务。')
    const base=new URL(config.url,location.origin)
    if(base.origin!==location.origin)throw new Error('查看器必须部署在同一服务')
    sessions[side]={src:viewerUrl(base.href,traceRange(props.unit,side)),status:'正在启动查看器…'}
    await nextTick()
    const target=frames[side].contentWindow
    await waitForViewer(target,location.origin,controller.signal)
    sessions[side].status='正在传入所选 Step Trace，解析进度见时间线…'
    const response=await fetch(`/api/jobs/${props.job.id}/trace-view/${side}?attempt=${encodeURIComponent(props.attempt||props.job.attempt)}`,{signal:controller.signal})
    if(gen!==generation || controller.signal.aborted)return
    await sendTrace(target,location.origin,response,`${props.job.name || props.job.id} · ${side.toUpperCase()}`)
    if(gen!==generation || controller.signal.aborted)return
    sessions[side].status='Trace 已交给查看器；解析进度见下方。加载后可点击“定位”重新聚焦。'
    focus()
  } catch(e) {if(gen===generation && !controller.signal.aborted){release(side);error.value=e.message}}
}
function move(delta) {eventIndex.value=Math.max(0,Math.min(events.value.length-1,eventIndex.value+delta));whole.value=false;focus()}
function enlarge() {padding.value*=2;focus()}
</script>
<template>
  <section class="trace-panel" :class="{expanded}" v-show="unit || expanded">
    <div class="toolbar"><button :disabled="!unit?.npu_kernels.length" @click="open('npu')">查看 NPU 调用时间线</button><button :disabled="!unit?.gpu_kernels.length" @click="open('gpu')">查看 GPU 调用时间线</button><button v-if="visible" @click="visible=false;expanded=false">收起时间线</button></div>
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <div v-show="visible">
      <p class="muted">两侧时间轴独立。只展示 Trace 实际记录的 CPU、runtime、flow 和 stream；没有 CPU 信息时无法还原调用链。定位为时间窗口，不代表已唯一选中事件，请核对轨道和名称。</p>
      <div class="toolbar"><label>当前事件<select v-model.number="eventIndex" @change="whole=false;focus()"><option v-for="(k,i) in events" :value="i" :key="k.kind+':'+k.index">{{ k.name }} · #{{ k.index }}<template v-if="k.name_position"> · 同名 {{ k.name_position.index }}/{{ k.name_position.total }}</template></option></select></label><button :disabled="eventIndex===0" @click="move(-1)">上一事件</button><button :disabled="eventIndex>=events.length-1" @click="move(1)">下一事件</button></div>
      <p v-if="current">{{ active.toUpperCase() }} · PID {{ current.pid }} / TID {{ current.tid ?? '未知' }} / stream {{ current.stream ?? '未知' }} · {{ current.time_us }} μs</p>
      <p v-if="current?.trace_present===false" class="notice">该 kernel 仅由 NPU 明细补充，无原始 Trace 事件。只能尝试查看附近时间范围。</p>
      <p v-else-if="current && current.trace_present==null" class="notice">旧结果没有原始事件定位信息，仅按时间范围查看。</p>
      <div class="toolbar"><button @click="whole=false;padding=1;focus()">定位当前事件</button><button @click="whole=true;padding=1;focus()">定位整个比较单元</button><button @click="enlarge">扩大上下文</button><button @click="expanded=!expanded">{{ expanded?'退出全屏':'全屏查看' }}</button><button @click="release(active);visible=false;expanded=false">释放当前 Trace 内存</button></div>
      <p class="muted" role="status">{{ sessions[active]?.status }}</p>
    </div>
    <template v-for="side in ['npu','gpu']" :key="side"><iframe v-if="sessions[side]" v-show="visible && active===side" :ref="el=>{frames[side]=el}" :src="sessions[side].src" :title="side.toUpperCase()+' Trace 时间线'" /></template>
  </section>
</template>
<style scoped>
.trace-panel{margin-top:20px;border-top:1px solid #dce5e2;padding-top:12px}.trace-panel iframe{width:100%;height:580px;border:1px solid #ccd7d4}.trace-panel select{max-width:650px}.trace-panel.expanded{position:fixed;inset:0;z-index:10000;background:white;margin:0;padding:16px;overflow:auto}.trace-panel.expanded iframe{height:calc(100vh - 300px);min-height:300px}
</style>
