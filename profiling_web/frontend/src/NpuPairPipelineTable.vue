<script setup>
import { computed, ref, watch } from 'vue'
import PipelineSummary from './PipelineSummary.vue'
const props = defineProps({job:Object, api:Function})
const page = ref(0), total = ref(0), rows = ref([]), error = ref(''), loading = ref(false), retry = ref(0)
const limit = 15
const search = ref(''), draft = ref(''), sort = ref('candidate')
const url = computed(()=>'/jobs/'+props.job.id+'/npu-pair-table?'+
  new URLSearchParams({view:'pipeline', search:search.value, sort:sort.value}))
function filter(){search.value=draft.value.trim(); page.value=0}
watch(()=>[url.value, props.job.attempt, page.value, retry.value], async (_,__,cleanup)=>{
  let current = true
  cleanup(()=>{current=false})
  rows.value=[]; error.value=''; loading.value=true
  try {
    const result = await props.api(url.value+'&offset='+page.value*limit+'&limit='+limit)
    if(current){rows.value=result.items; total.value=result.total}
  } catch(e){if(current)error.value=e.message} finally {if(current)loading.value=false}
},{immediate:true})
</script>
<template>
  <form class="toolbar" @submit.prevent="filter">
    <input v-model="draft" placeholder="搜索 OP Type / Core Type" aria-label="搜索算子流水" maxlength="200">
    <select v-model="sort" @change="filter" aria-label="算子流水排序"><option value="candidate">对比耗时降序</option><option value="baseline">基线耗时降序</option></select>
    <button>筛选</button>
  </form>
  <p v-if="error" role="alert">{{ error }} <button @click="retry++">重试</button></p>
  <p v-else-if="loading">正在读取算子流水…</p>
  <p v-else-if="!total" class="muted">此结果没有可显示的算子流水。</p>
  <div v-else class="table-scroll"><table><thead><tr><th>OP Type</th><th>Core Type</th><th>基线 · 流水构成</th><th>对比侧 · 流水构成</th></tr></thead>
    <tbody><tr v-for="(r,i) in rows" :key="i"><td>{{ r['OP Type'] }}</td><td>{{ r['Core Type'] || '未标记' }}</td>
      <td><PipelineSummary :data="r['基线']" compact/></td><td><PipelineSummary :data="r['对比']" compact/></td></tr></tbody></table></div>
  <div class="pagination"><span>共 {{ total }} 个算子 · 第 {{ page+1 }} 页</span><button :disabled="loading || !page" @click="page--">上一页</button><button :disabled="loading || (page+1)*limit>=total" @click="page++">下一页</button></div>
</template>
<style scoped>
td,th{white-space:nowrap;vertical-align:middle}
</style>
