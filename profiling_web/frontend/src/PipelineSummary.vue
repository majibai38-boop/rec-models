<script setup>
import {computed} from 'vue'
import {fmt} from './operatorDetailView.js'
const props = defineProps({data:Object, compact:Boolean})
const items = computed(() => [...(props.data?.categories || [])].sort((a,b)=>b.time_us-a.time_us))
</script>
<template>
  <div class="pipeline-summary">
    <template v-if="data">
      <p v-if="!compact" class="muted">NPU 流水瓶颈初筛 · 按 kernel 耗时分布（非墙钟时间）。仅为候选，不表示 HBM 带宽或算力已饱和。Unknown 计入分母。</p>
      <div class="pipeline-bar" role="img" :aria-label="items.map(i=>i.label+' '+fmt(i.time_pct)+'%').join('；')">
        <span v-for="i in items" :key="i.category" :class="i.category" :style="{width:(i.time_pct || 0)+'%'}" :title="i.label+' '+fmt(i.time_pct)+'%'" />
      </div>
      <details v-if="compact"><summary>{{ items[0]?.label || '无 kernel' }} {{ fmt(items[0]?.time_pct) }}% · 耗时分布</summary><p v-for="i in items" :key="i.category">{{ i.label }}：{{ fmt(i.time_pct) }}% · {{ fmt(i.time_us) }} μs · {{ i.count }} 次</p></details>
      <div v-else class="pipeline-legend"><span v-for="i in items" :key="i.category"><i :class="i.category"/>{{ i.label }} {{ fmt(i.time_pct) }}%<small>{{ fmt(i.time_us) }} μs · {{ i.count }} 次</small></span></div>
    </template>
    <small v-else class="muted">旧结果未生成流水初筛；复用输入重新分析后可查看。</small>
  </div>
</template>
<style scoped>
.pipeline-summary{min-width:150px;margin:8px 0}.pipeline-bar{display:flex;height:8px;overflow:hidden;border-radius:4px;background:#edf0f2}.pipeline-bar span{height:100%}.compute{background:#487cbd}.memory{background:#249787}.scalar{background:#9972b5}.mixed{background:#cb8a3d}.no_bound{background:#b5a54f}.unknown{background:#969da8}.pipeline-legend{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0}.pipeline-legend i{display:inline-block;width:10px;height:10px;margin-right:6px;border-radius:2px}.pipeline-legend small{display:block;margin-left:16px}.pipeline-summary p{margin:6px 0}.pipeline-summary summary{cursor:pointer;font-size:12px}
</style>
