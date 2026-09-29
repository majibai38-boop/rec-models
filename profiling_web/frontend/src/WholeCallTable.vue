<script setup>
defineProps({ rows: Array, details: Boolean })
defineEmits(['group', 'detail'])
const num = v => v === '' || v == null ? '—' : Number(v).toLocaleString('en-US', { maximumFractionDigits: 3 })
const ratioClass = v => v === '' || v == null ? '' : Number(v) >= 2 ? 'ratio-dark' : Number(v) >= 1 ? 'ratio-light' : 'ratio-yellow'
</script>

<template>
  <div class="table-scroll"><table>
    <thead><tr>
      <th>整体调用</th><th>NPU / GPU 组成</th>
      <th v-if="!details">可比 / 待核实调用数</th><th v-else>CPU 对齐状态</th>
      <th>{{ details ? '完整调用耗时' : '可比较调用耗时' }} μs<small>NPU / GPU</small></th>
      <th>Comparable Ratio</th><th v-if="!details">Total Time Ratio</th><th>比较状态 / 范围</th><th>证据</th>
    </tr></thead>
    <tbody><tr v-for="r in rows" :key="r['Call ID'] || r['Group ID']">
      <td><button class="type-link" @click="$emit(details ? 'detail' : 'group', r)">{{ r['Whole OP'] }}</button>
        <small>{{ r['Call ID'] || r['Group ID'] }}</small><small>Device {{ r.Device_id || '未提供' }}</small></td>
      <td class="composition">{{ r['NPU Types'] }}<small>{{ r['GPU Composition'] }}</small></td>
      <td v-if="!details">{{ num(r['Comparable Call Count']) }} / {{ num(r['Pending Call Count']) }}
        <small>关联 {{ num(r['Observed Call Count']) }} · 组内重叠未重复计入 {{ num(r['Overlapping Call Count']) }}</small></td>
      <td v-else>{{ r['CPU Alignment Status'] }}</td>
      <td>{{ num(r[details ? 'NPU Total Time(us)' : 'Comparable NPU Time(us)']) }}<small>{{ num(r[details ? 'GPU Total Time(us)' : 'Comparable GPU Time(us)']) }}</small>
        <small v-if="details">GPU 内存 {{ num(r['GPU Memory Time(us)']) }} μs</small></td>
      <td><span :class="['ratio', ratioClass(r['Comparable Ratio(GPU/NPU)'])]">{{ num(r['Comparable Ratio(GPU/NPU)']) }}</span></td>
      <td v-if="!details"><span :class="['ratio', ratioClass(r['Total Time Ratio(GPU/NPU)'])]">{{ num(r['Total Time Ratio(GPU/NPU)']) }}</span></td>
      <td class="conclusion">{{ r[details ? 'Kernel Comparison Status' : 'Comparison Status'] }}
        <small>{{ r[details ? 'Issues' : 'Population Status'] }}</small>
        <small v-if="details && r['Overlap Call IDs'] !== '[]'">与其他调用重叠：{{ r['Overlap Call IDs'] }}</small></td>
      <td><button @click="$emit(details ? 'detail' : 'group', r)">{{ details ? '事件证据' : '调用明细' }}</button>
        <details v-if="!details"><summary>完整范围</summary><p>NPU 调用 {{ num(r['NPU Population Call Count']) }} / GPU 调用 {{ num(r['GPU Population Call Count']) }}</p><p>NPU {{ num(r['NPU Population Time(us)']) }} / GPU {{ num(r['GPU Population Time(us)']) }} μs</p></details></td>
    </tr><tr v-if="!rows.length"><td colspan="8" class="empty-row">没有对应整体记录。旧任务需重新分析才能生成新视图；没有记录不代表没有工作。</td></tr></tbody>
  </table></div>
</template>
