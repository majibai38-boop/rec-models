<script setup>
import PipelineSummary from './PipelineSummary.vue'
defineProps({ rows: Array, detailed: Boolean })
defineEmits(['type', 'whole', 'report', 'manual', 'detail'])
const fmt = (v, n = 2) => v == null || v === '' ? '—' : Number(v).toLocaleString('en-US', { maximumFractionDigits: n })
const color = v => v == null ? '' : v >= 2 ? 'ratio-dark' : v >= 1 ? 'ratio-light' : 'ratio-yellow'
const ratioHint = (r, comparable) => comparable
  ? `已分配 GPU 时间 ${fmt(r.gpu_total_time_us)} μs / 可比 NPU 时间 ${fmt(r.npu_comparable_time_us)} μs；缺少可比工作时不输出比值`
  : `已分配 GPU 时间 ${fmt(r.gpu_total_time_us)} μs / NPU 类型总时间 ${fmt(r.npu_total_time_us)} μs；不是完整同类 GPU 工作量证明`
</script>
<template>
  <div class="table-scroll"><table>
    <thead><tr><th>类型 / 设备</th><th v-if="detailed">NPU / GPU 数量</th><th>NPU 时间 μs<small>全任务时间占比</small></th><th>已分配 GPU 时间 μs</th><th title="GPU/NPU；大于 1 表示可比范围内 NPU 耗时更少">可比比值</th><th>类型全量口径比值</th><th>可比时间覆盖率<small v-if="detailed">详细：数量 / 时间 %</small></th><th>NPU 流水初筛<small>候选 · 耗时分布</small></th><th>结论与整体比较</th></tr></thead>
    <tbody><tr v-for="r in rows" :key="[r.device_id,r.op_type,r.core_type].join('|')">
      <td><button class="type-link" :disabled="r.row_kind !== 'npu_type'" @click="$emit('type', r)">{{ r.op_type }}</button><small>{{ r.device_id }} / {{ r.core_type }}</small><button v-if="r.row_kind==='npu_type'" class="type-link" @click="$emit('detail',r)">详细对比 →</button><small v-if="detailed && (r.fused_npu_type !== r.op_type || r.fused_gpu_type !== r.op_type)">{{ r.fused_npu_type }} ↔ {{ r.fused_gpu_type }}</small><button v-if="r.row_kind==='npu_type' && (r.coverage.comparable.time_pct < 100 || r.manual_count)" class="type-link" @click="$emit('manual',r)">{{ r.manual_count ? '人工关联 ×'+r.manual_count : '补充关联' }}</button></td>
      <td v-if="detailed">{{ fmt(r.npu_kernel_count,0) }} / {{ fmt(r.gpu_kernel_count,0) }}<small>GPU 内存活动 {{ fmt(r.gpu_memory_activity_count,0) }}</small></td>
      <td>{{ fmt(r.npu_total_time_us) }}<small>{{ fmt(r.npu_ratio_pct) }}%</small></td><td>{{ fmt(r.gpu_total_time_us) }}</td>
      <td :title="ratioHint(r,true)"><span :class="['ratio',color(r.comparable_ratio_gpu_npu)]">{{ fmt(r.comparable_ratio_gpu_npu,3) }}</span><small v-if="r.comparable_ratio_gpu_npu == null">无可比值</small></td>
      <td :title="ratioHint(r,false)"><span :class="['ratio',color(r.total_time_ratio_gpu_npu)]">{{ fmt(r.total_time_ratio_gpu_npu,3) }}</span></td>
      <td><template v-if="detailed"><div v-for="k in ['direct','comparable','accounted']" :key="k" class="coverage-line"><span>{{ k }}</span>{{ fmt(r.coverage[k].count_pct,1) }} / {{ fmt(r.coverage[k].time_pct,1) }}</div></template><template v-else>{{ fmt(r.coverage.comparable.time_pct,1) }}%</template></td>
      <td><PipelineSummary v-if="r.row_kind==='npu_type'" :data="r.pipeline" compact/><span v-else>—</span></td><td class="conclusion">{{ r.conclusion || '—' }}<small v-if="detailed">{{ r.cpu_alignment_status }} · {{ r.kernel_comparison_status }}</small><small>{{ r.whole_status }}</small><button v-if="r.whole_group_ids?.length" class="type-link" @click="$emit('whole',r)">{{ r.related_whole_ops }} →</button><button v-else-if="r.related_whole_ops" class="type-link" @click="$emit('report')">查看排查报告</button></td>
    </tr><tr v-if="!rows.length"><td colspan="9" class="empty-row">没有符合当前筛选条件的类型。请清除筛选后重试。</td></tr></tbody>
  </table></div>
</template>
