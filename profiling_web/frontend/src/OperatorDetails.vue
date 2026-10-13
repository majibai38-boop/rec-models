<script setup>
import {computed, ref, watch, onBeforeUnmount, nextTick} from 'vue'
import Chart from './Chart.vue'
import PipelineSummary from './PipelineSummary.vue'
import PipelineEvidence from './PipelineEvidence.vue'
import {csvCell} from './viewState.js'
import {fmt, ratioClass, tensorText, numel} from './operatorDetailView.js'
const props = defineProps({job:Object, scope:Object, state:Object, api:Function, detailed:Boolean})
const emit = defineEmits(['update', 'whole', 'coverage', 'upgrade', 'types'])
const data=ref(null), chartData=ref([]), chartTotal=ref(0), units=ref(null), selected=ref(null), error=ref('')
const groupLoading=ref(false), unitLoading=ref(false), exporting=ref(false), dialog=ref(null)
const unitSection=ref(null)
let scrollToUnits=false
let groupSequence=0, unitSequence=0, detailSequence=0, disposed=false
const inputSearch=ref(props.state.search || '')
const change = values => emit('update', {...props.state,...values})
function url(params={}) {return '/jobs/'+props.job.id+'/operator-details?'+new URLSearchParams({...props.scope,attempt:props.state.attempt||props.job.attempt,...params})}
async function loadGroups() {
  const seq=++groupSequence
  data.value=null;chartData.value=[];error.value=''
  if(!props.scope?.op_type)return
  groupLoading.value=true
  try {
    const params={search:props.state.search,sort:props.state.sort}
    const [page,chart]=await Promise.all([props.api(url({...params,offset:props.state.page*20,limit:20})),props.api(url({...params,sort:'npu_time',limit:200}))])
    if(disposed||seq!==groupSequence)return
    data.value=page;chartData.value=chart.items;chartTotal.value=chart.total
  } catch(e) {if(!disposed&&seq===groupSequence)error.value=e.message}
  finally {if(seq===groupSequence)groupLoading.value=false}
}
async function loadUnits() {
  const seq=++unitSequence; units.value=null
  if(!props.scope?.op_type||!props.state.group)return
  unitLoading.value=true
  try {
    const result=await props.api(url({view:'units',group_id:props.state.group,sort:'position',offset:props.state.unitPage*20,limit:20}))
    if(!disposed&&seq===unitSequence){units.value=result;await nextTick();if(scrollToUnits){unitSection.value?.scrollIntoView({block:'start'});scrollToUnits=false}}
  }catch(e){if(!disposed&&seq===unitSequence)error.value=e.message}
  finally{if(seq===unitSequence)unitLoading.value=false}
}
async function loadDetail() {
  const seq=++detailSequence; selected.value=null; dialog.value?.close()
  if(!props.scope?.op_type||!props.state.unit)return
  try {
    const result=await props.api(url({view:'units',unit_id:props.state.unit,limit:1}))
    if(disposed||seq!==detailSequence)return
    selected.value=result.items[0]||null
    if(!selected.value){error.value='该比较单元不在当前类型/结果版本中';return}
    await nextTick();dialog.value?.showModal()
  }catch(e){if(!disposed&&seq===detailSequence)error.value=e.message}
}
const scopeKey=computed(()=>JSON.stringify([props.job.id,props.job.attempt,props.scope]))
watch([()=>scopeKey.value,()=>props.state.attempt,()=>props.state.search,()=>props.state.sort,()=>props.state.page],loadGroups,{immediate:true})
watch([()=>scopeKey.value,()=>props.state.attempt,()=>props.state.group,()=>props.state.unitPage],loadUnits,{immediate:true})
watch([()=>scopeKey.value,()=>props.state.attempt,()=>props.state.unit],loadDetail,{immediate:true})
watch(()=>props.state.search,v=>{inputSearch.value=v})
onBeforeUnmount(()=>{disposed=true;dialog.value?.close()})
function choose(group) {scrollToUnits=true;if(props.state.group===group.group_id){unitSection.value?.scrollIntoView({block:'start'});scrollToUnits=false}change({group:group.group_id,unitPage:0,unit:''})}
function chartClick(p) {const row=chartData.value.find(r=>r.group_id===p.data?.group_id);if(row)choose(row)}
const currentGroup=computed(()=>chartData.value.find(r=>r.group_id===props.state.group)||data.value?.items.find(r=>r.group_id===props.state.group))
const summary=computed(()=>data.value?.summary || {})
const coverage=computed(()=>summary.value.npu_time_us>0?100*summary.value.comparable_npu_time_us/summary.value.npu_time_us:null)
const ratio=computed(()=>summary.value.comparable_npu_time_us>0?summary.value.comparable_gpu_time_us/summary.value.comparable_npu_time_us:null)
const inputCount=computed(()=>Math.min(1000,Math.max(1,...chartData.value.map(r=>r.npu_metadata[0]?.input_shapes?.length||0))))
const plotted=computed(()=>chartData.value.filter(r=>r.ratio!=null&&numel(r.npu_metadata[0],props.state.input)!=null))
const tooltip={renderMode:'richText',confine:true}
const bars=computed(()=>{
  const rows=chartData.value.slice(0,15).reverse()
  return {tooltip:{...tooltip,trigger:'axis'},legend:{data:['NPU 可比均值','GPU 可比均值']},grid:{left:125,right:25,top:40,bottom:40},
    xAxis:{type:'value',name:'μs'},yAxis:{type:'category',data:rows.map(r=>'shape '+r.group_id.slice(0,6))},
    series:[['NPU 可比均值','npu_mean_us','#167d78'],['GPU 可比均值','gpu_mean_us','#7489ba']].map(([name,field,color])=>({name,type:'bar',itemStyle:{color},
      data:rows.map(r=>({value:r[field],group_id:r.group_id,name:tensorText(r.npu_metadata[0])}))}))}
})
const scatter=computed(()=>({tooltip:{...tooltip,formatter:p=>`${p.data.name}\n元素数 ${fmt(p.value[0])}\nGPU/NPU ${fmt(p.value[1])}`},
  grid:{left:65,right:40,top:30,bottom:60},xAxis:{type:'value',name:`NPU kernel[0].input[${props.state.input}] 元素数`,nameLocation:'middle',nameGap:35},yAxis:{type:'value',name:'GPU/NPU'},
  series:[{type:'scatter',markLine:{silent:true,data:[{yAxis:1},{yAxis:2}]},data:plotted.value.map(r=>({name:tensorText(r.npu_metadata[0]),group_id:r.group_id,
    value:[numel(r.npu_metadata[0],props.state.input),r.ratio],symbolSize:Math.min(32,8+Math.sqrt(r.npu_time_us)),
    itemStyle:{color:r.ratio>=2?'#176741':r.ratio>=1?'#79b891':'#dbb53f'}}))}]}))
async function exportGroups() {
  exporting.value=true;error.value=''
  const base=url({search:props.state.search,sort:props.state.sort,limit:200}), rows=[]
  try {
    for(let offset=0;;offset+=200){const page=await props.api(base+'&offset='+offset);if(page.total>100000)throw Error('超过 10 万组，请缩小筛选');rows.push(...page.items);if(!page.has_more)break}
    const fields=[...new Set(rows.flatMap(r=>Object.keys(r)))], lines=[fields,...rows.map(r=>fields.map(f=>r[f]))]
    const link=document.createElement('a'), target=URL.createObjectURL(new Blob(['\uFEFF'+lines.map(r=>r.map(csvCell).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}))
    link.href=target;link.download='operator-shape-groups.csv';link.click();setTimeout(()=>URL.revokeObjectURL(target),1000)
  }catch(e){error.value=e.message}finally{exporting.value=false}
}
</script>
<template>
  <section class="operator-details">
    <div class="panel-heading"><h2>算子详情<span v-if="scope"> · {{ scope.op_type }} / {{ scope.device_id || '未提供设备' }} / {{ scope.core_type }}</span></h2><button @click="emit('types')">返回类型对比</button></div>
    <p v-if="!scope?.op_type" class="notice">请在类型对比中选择一个 NPU 类型，点击“详细对比”。</p>
    <template v-else>
      <p class="muted">结果版本 {{ state.attempt || job.attempt }} · 事件时长之和（非墙钟时间）。按完整元数据、模块、两侧 kernel 组成分组；不按耗时或元素数推断映射。</p>
      <p v-if="state.attempt && state.attempt!==job.attempt" class="notice">此链接指向较早结果，请重新选择当前版本中的分组。<button @click="change({attempt:job.attempt,group:'',unit:'',page:0,unitPage:0})">查看当前结果版本</button></p>
      <PipelineSummary v-if="data?.supported" :data="summary.pipeline"/>
      <div class="toolbar"><button @click="emit('coverage',scope)">覆盖率诊断</button><button @click="emit('whole',scope)">查看整体比较</button></div>
      <p v-if="error" class="error" role="alert">{{ error }} <button @click="loadGroups();loadUnits();loadDetail()">重试</button></p>
      <p v-if="groupLoading" role="status">正在查询 shape 分组…</p>
      <div v-if="data && !data.supported" class="notice">旧结果缺少完整元数据索引，可查看现有覆盖率诊断/逐 kernel 映射；完整详情需重新分析，旧结果保持不变。<button @click="emit('upgrade')">复用输入，新建分析</button></div>
      <template v-if="data?.supported">
        <div class="detail-metrics"><div>比较单元 <strong>{{ fmt(summary.comparable_count) }} / {{ fmt(summary.count) }}</strong><small>可比 / 全部，不等于 CPU 调用数</small></div><div>NPU 总耗时 <strong>{{ fmt(summary.npu_time_us) }} μs</strong></div><div>独立可比 GPU 时间 <strong>{{ fmt(summary.comparable_gpu_time_us) }} μs</strong></div><div>可比时间覆盖率 <strong>{{ fmt(coverage) }}%</strong></div><div>可比 GPU/NPU <strong :class="['ratio',ratioClass(ratio)]">{{ fmt(ratio) }}</strong></div></div>
        <p class="notice">此页只对独立比较单元计算性能比；共享/跨类型 GPU 时间不分摊，因此可比范围可能小于类型表。均值、P50/P95 与比值均使用同一批可比单元，未映射 GPU 不补零。人工确认保留来源标记。</p>
        <form class="toolbar" @submit.prevent="change({search:inputSearch,page:0,group:'',unit:''})"><input v-model="inputSearch" aria-label="搜索 shape dtype 或算子" placeholder="搜索 shape / dtype / 模块 / kernel" maxlength="200"><select :value="state.sort" aria-label="详情分组排序" @change="change({sort:$event.target.value,page:0})"><option value="npu_time">NPU 总耗时降序</option><option value="ratio">可比比值降序</option></select><button>筛选分组</button><button type="button" @click="inputSearch='';change({search:'',page:0,group:'',unit:''})">清除筛选</button><button type="button" :disabled="exporting" @click="exportGroups">导出分组 CSV</button></form>
        <div class="detail-charts"><section><h3>shape 耗时对比 · Top 15</h3><Chart :option="bars" @select="chartClick"/></section><section><h3>输入规模与性能比</h3><label>输入位置 <select :value="state.input" @change="change({input:Number($event.target.value)})"><option v-for="n in inputCount" :key="n" :value="n-1">NPU kernel[0].input[{{ n-1 }}]</option></select></label><Chart :option="scatter" @select="chartClick"/></section></div>
        <p class="muted small">图表基于筛选后 NPU 总耗时最高的 {{ chartData.length }} / {{ chartTotal }} 组；散点 {{ plotted.length }} 组，其余缺少有效输入或可比值。气泡大小表示 NPU 总耗时。点击图形查看逐次记录，完整数据见下表/导出。</p>
        <div class="table-scroll"><table><thead><tr><th>分组 / 算子</th><th>NPU kernel 输入 / 输出</th><th>GPU CPU op 输入<small>不是 GPU kernel 参数</small></th><th>可比 / 全部单元</th><th>可比均值 μs<small>NPU / GPU</small></th><th>GPU/NPU</th><th>NPU 总时间 μs / 覆盖率</th><th>NPU 流水初筛</th><th v-if="detailed">P50 / P95 μs</th></tr></thead><tbody>
          <tr v-for="r in data.items" :key="r.group_id" :class="{'selected-detail-row':state.group===r.group_id}"><td><button class="type-link" @click="choose(r)">shape {{ r.group_id.slice(0,6) }} →</button><small>{{ r.npu_cpu.map(c=>c.name).join(' + ') || '未知 CPU op' }}</small><small>{{ r.granularity==='kernel'?'单 kernel':'执行组' }} · {{ r.npu_kernel_count }} NPU kernels</small><small v-if="r.manual_count">人工确认 {{ r.manual_count }} 个单元</small><details><summary>组成与模块</summary><p class="metadata-text">{{ r.gpu_names.join('\n') || 'GPU 未对齐' }}</p><p>{{ r.npu_cpu.map(c=>c.module).join(' / ') || '未知模块' }}</p></details></td>
            <td><div v-for="(m,i) in r.npu_metadata" :key="i"><small>{{ m.name }}</small><p class="metadata-text">{{ tensorText(m) }}</p><details><summary>输出 shape / dtype</summary><p class="metadata-text">{{ tensorText(m,'output') }}</p></details></div></td>
            <td><div v-for="(c,i) in r.gpu_cpu" :key="i"><small>{{ c.name }}</small><p class="metadata-text">{{ tensorText(c) }}</p></div><span v-if="!r.gpu_cpu.length">未采集 / 未对齐</span></td><td>{{ r.comparable_count }} / {{ r.count }}<small>{{ r.statuses.join('；') }}</small></td><td>{{ fmt(r.npu_mean_us) }} / {{ fmt(r.gpu_mean_us) }}</td><td><span :class="['ratio',ratioClass(r.ratio)]">{{ fmt(r.ratio) }}</span></td><td>{{ fmt(r.npu_time_us) }} / {{ fmt(r.coverage_pct) }}%</td><td><PipelineSummary :data="r.pipeline" compact/></td><td v-if="detailed">NPU {{ fmt(r.npu_p50_us) }} / {{ fmt(r.npu_p95_us) }}<small>GPU {{ fmt(r.gpu_p50_us) }} / {{ fmt(r.gpu_p95_us) }}</small></td></tr>
          <tr v-if="!data.items.length"><td colspan="9">没有符合筛选条件的分组。</td></tr></tbody></table></div>
        <div class="pagination"><span>{{ data.total }} 组 · 第 {{ state.page+1 }} 页</span><button :disabled="state.page===0" @click="change({page:state.page-1})">上一页分组</button><button :disabled="!data.has_more" @click="change({page:state.page+1})">下一页分组</button></div>
        <section v-if="state.group" ref="unitSection"><h3>逐次比较 · shape {{ state.group.slice(0,6) }}</h3><p v-if="currentGroup" class="metadata-text">{{ tensorText(currentGroup.npu_metadata[0]) }}</p><p class="muted">类型单元序号基于本结果版本完整类型集合，模板序号基于当前分组；筛选/翻页不重新编号。一个单元可能包含多个 CPU 调用或 kernel。</p><p v-if="unitLoading" role="status">正在加载逐次记录…</p><div class="table-scroll" v-if="units"><table><thead><tr><th>类型单元 / 模板序号</th><th>NPU / GPU 事件数量</th><th>NPU / GPU 时间 μs</th><th>GPU/NPU</th><th>NPU 流水初筛</th><th>状态 / 证据</th></tr></thead><tbody><tr v-for="u in units.items" :key="u.id"><td>{{ u.position.index }}/{{ u.position.total }} · {{ u.template_position.index }}/{{ u.template_position.total }}</td><td>{{ u.npu_kernels.length }} / {{ u.gpu_kernels.length }}</td><td>{{ fmt(u.npu_time_us) }} / {{ fmt(u.gpu_time_us) }}</td><td><span :class="['ratio',ratioClass(u.ratio)]">{{ fmt(u.ratio) }}</span></td><td><PipelineSummary :data="u.pipeline" compact/></td><td>{{ u.status }}<small v-if="u.manual">人工确认（非自动验证）</small><button @click="change({unit:u.id})">双侧详情</button></td></tr><tr v-if="!units.items.length"><td colspan="6">此分组没有记录。</td></tr></tbody></table></div><div class="pagination" v-if="units"><span>{{ units.total }} 个比较单元</span><button :disabled="state.unitPage===0" @click="change({unitPage:state.unitPage-1})">上一页调用</button><button :disabled="!units.has_more" @click="change({unitPage:state.unitPage+1})">下一页调用</button></div></section>
      </template>
    </template>
    <dialog ref="dialog" aria-label="双侧算子详情" class="operator-dialog" @cancel.prevent="change({unit:''})"><template v-if="selected"><div class="panel-heading"><h2>双侧算子详情 · {{ selected.id }}</h2><button autofocus @click="change({unit:''})">关闭详情</button></div><p>{{ selected.status }} · {{ selected.granularity==='kernel'?'单 kernel':'执行组' }} · GPU/NPU <span :class="['ratio',ratioClass(selected.ratio)]">{{ fmt(selected.ratio) }}</span></p><p class="muted">{{ selected.basis.join(' / ') }}<span v-if="selected.manual"> · 人工确认：{{ selected.manual_reasons.join('；') }}</span></p><div class="detail-sides"><section v-for="side in ['npu','gpu']" :key="side"><h3>{{ side.toUpperCase() }} · {{ fmt(selected[side+'_time_us']) }} μs</h3><h4>CPU 比较范围</h4><p v-if="!selected[side+'_scopes'].length" class="muted">未记录独立 CPU 比较范围；以下 owner 不自动代表比较 scope。</p><article v-for="c in selected[side+'_scopes']" :key="c.index"><strong>{{ c.name }} · CPU #{{ c.index }}</strong><p>{{ c.module }}</p><p class="metadata-text">{{ tensorText(c) }}</p><p class="metadata-text">{{ tensorText(c,'output') }}</p></article><details><summary>实际 kernel owner</summary><article v-for="c in selected[side+'_owners']" :key="c.index"><strong>{{ c.name }} · CPU #{{ c.index }}</strong><p>{{ c.path.join(' → ') }}</p><p class="metadata-text">{{ tensorText(c) }}</p><p class="metadata-text">{{ tensorText(c,'output') }}</p></article></details><h4>设备事件</h4><article v-for="k in selected[side+'_kernels']" :key="k.kind+':'+k.index"><strong>{{ k.kind }} #{{ k.index }} · {{ k.name }}</strong><p>{{ fmt(k.time_us) }} μs · stream {{ k.stream || '未知' }}</p><PipelineEvidence v-if="side==='npu'" :data="k.pipeline"/><small>数据来源：{{ k.source==='kernel_details'?'NPU kernel_details':'未采集 kernel 输入输出；CPU 元数据见上方' }}</small><p class="metadata-text">{{ tensorText(k) }}</p><p class="metadata-text">{{ tensorText(k,'output') }}</p></article><p v-if="!selected[side+'_kernels'].length">没有已关联设备事件；耗时未知，不视为零。</p></section></div></template></dialog>
  </section>
</template>
<style scoped>
.detail-metrics{display:flex;flex-wrap:wrap;gap:24px;margin:20px 0}.detail-metrics strong,.detail-metrics small{display:block}.detail-charts,.detail-sides{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}.metadata-text{white-space:pre-wrap;overflow-wrap:anywhere;margin:6px 0}.selected-detail-row{background:#edf6f1}.operator-dialog{width:min(1200px,94vw);max-height:90vh;overflow:auto;border:1px solid #ccd7d4;border-radius:12px;padding:24px;color:#20332f;background:#fff}.operator-dialog::backdrop{background:#102c3266}.detail-sides article{padding:12px 0;border-bottom:1px solid #dce5e2;overflow-wrap:anywhere}.detail-charts section{min-width:0}@media(max-width:1100px){.detail-charts{grid-template-columns:minmax(0,1fr)}}@media(max-width:700px){.detail-sides{grid-template-columns:minmax(0,1fr)}}
</style>
