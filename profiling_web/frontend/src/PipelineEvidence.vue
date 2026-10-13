<script setup>
import {fmt} from './operatorDetailView.js'
defineProps({data:Object})
</script>
<template>
  <details v-if="data" class="pipeline-evidence">
    <summary>NPU 流水初筛：{{ data.label }}（{{ data.confidence==='heuristic'?'启发式候选':'证据不足' }}）</summary>
    <p>{{ data.reason }}</p><p>下一步：{{ data.next_check }}</p>
    <section v-for="side in data.sides" :key="side.side">
      <strong>{{ side.side.toUpperCase() }} · {{ side.label }}</strong><p>{{ side.reason }}</p>
      <table><thead><tr><th>指标</th><th>周期占比</th><th>原始值</th></tr></thead><tbody><tr v-for="(value,key) in side.ratios" :key="key"><td>{{ key }}</td><td>{{ value==null?'未采集 / 无效':fmt(value*100)+'%' }}</td><td>{{ side.raw_values[key] ?? '缺列' }}</td></tr></tbody></table>
      <p v-if="data.sides.length>1">下一步：{{ side.next_check }}</p>
    </section>
    <small>参考阈值 {{ fmt(data.threshold*100) }}%；跨类别差距不足 {{ fmt(data.dominance_margin*100) }} 个百分点保留混合候选。不同流水占比不可相加；未确认 Roofline 瓶颈。</small>
  </details>
</template>
<style scoped>
.pipeline-evidence{margin:12px 0;padding:10px;border:1px solid #dce5e2;border-radius:6px}.pipeline-evidence summary{cursor:pointer}.pipeline-evidence table{font-size:12px;width:100%}.pipeline-evidence td{overflow-wrap:anywhere}.pipeline-evidence small{display:block;margin-top:8px}
</style>
