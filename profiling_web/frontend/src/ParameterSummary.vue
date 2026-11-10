<script setup>
import { ref } from 'vue'
defineProps({ data: Object })
const selected = ref('npu')
const count = v => v == null ? '未识别' : `${Number(v).toLocaleString('zh-CN')}（${(v/1e6).toFixed(3)} M）`
const metrics = [['candidate_parameters','候选参数合计'],['dense_candidates','稠密权重候选'],['embedding_candidates','Embedding表候选'],['ambiguous_matrix_elements','待确认矩阵右输入元素（不计入合计）']]
const kinds = {dense:'稠密',embedding:'Embedding',ambiguous:'待确认矩阵'}
</script>

<template>
  <section class="panel parameter-summary" aria-label="模型参数量画像">
    <h2>模型参数量 · Trace Shape估算</h2>
    <p v-if="!data" class="muted">旧报告尚无参数画像，请复用输入重新分析。</p>
    <template v-else>
      <p class="muted small">{{ data.note }}</p>
      <div class="table-scroll"><table>
        <thead><tr><th>指标</th><th>NPU</th><th>GPU</th></tr></thead>
        <tbody><tr v-for="[key,label] in metrics" :key="key"><td>{{ label }}</td><td>{{ count(data.npu[key]) }}</td><td>{{ count(data.gpu[key]) }}</td></tr></tbody>
      </table></div>
      <p class="muted small">{{ data.npu.note }}</p>
      <details>
        <summary>查看候选 Shape 明细</summary>
        <label>查看设备 <select v-model="selected" aria-label="参数量设备"><option value="npu">NPU</option><option value="gpu">GPU</option></select></label>
        <p class="muted small">PID {{ data[selected].selection.pid }} · Step {{ data[selected].selection.step || '全范围' }} · {{ data[selected].groups.length }} 组 · 缺少有效权重Shape的候选调用 {{ data[selected].missing_shape_calls }} 次。最多展示前100组，完整明细保存在服务器本次报告的 parameter_estimates.json 中。</p>
        <div class="table-scroll"><table>
          <thead><tr><th>类别</th><th>算子 / 角色</th><th>Shape</th><th>dtype</th><th>模块</th><th>元素数</th><th>调用次数</th><th>去重依据</th></tr></thead>
          <tbody><tr v-for="(row,i) in data[selected].groups.slice(0,100)" :key="selected+':'+i">
            <td>{{ kinds[row.kind] }}</td><td>{{ row.op }} / {{ row.role }}</td><td>{{ JSON.stringify(row.shape) }}</td><td>{{ row.dtype }}</td><td>{{ row.module || '未记录' }}</td><td>{{ count(row.numel) }}</td><td>{{ row.calls }}</td><td>{{ row.dedup_basis==='module_shape'?'模块+Shape':'仅Shape' }}</td>
          </tr></tbody>
        </table></div>
      </details>
    </template>
  </section>
</template>

<style scoped>
.parameter-summary {margin-top:18px;}
summary {cursor:pointer;}
select {margin:12px;}
</style>
