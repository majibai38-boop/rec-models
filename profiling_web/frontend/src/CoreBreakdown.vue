<script setup>
defineProps({ data: Object })
const fmt = n => n == null ? '—' : Number(n).toLocaleString('en-US', {maximumFractionDigits: 3})
const percent = n => n == null ? '—' : `${fmt(n)}%`
const labels = {Cube: 'Cube', Vector: 'Vector', Unknown: '未知', Mixed: '混合 Cube/Vector', Unmapped: '未映射'}
</script>

<template>
  <section class="panel core-breakdown" aria-label="Cube 和 Vector 总览">
    <h2>Cube / Vector 总览</h2>
    <p v-if="!data" class="muted">此报告尚无 Cube/Vector 统计，请复用输入重新分析。</p>
    <template v-else>
      <p class="muted small">NPU 按 aicore_time 分类：0 为 Vector，大于 0 为 Cube，缺失或无效为未知。耗时采用 kernel Duration，计数器单独汇总。</p>
      <div class="core-columns">
        <article v-for="side in ['npu', 'gpu_reference']" :key="side">
          <h3>{{ side === 'npu' ? 'NPU' : 'GPU 类 Cube / Vector（参考）' }}</h3>
          <p class="muted small">{{ fmt(data[side].total_count) }} 个 kernel · {{ fmt(data[side].total_time_us) }} μs</p>
          <div class="class-bar" aria-hidden="true"><span v-for="g in data[side].groups" :key="g.category" :class="g.category" :style="{width: `${g.time_pct || 0}%`}" :title="`${labels[g.category]} ${percent(g.time_pct)}`" /></div>
          <div class="table-scroll"><table>
            <thead><tr><th>类别</th><th>数量 / 占比</th><th>耗时 μs</th><th>时间占比</th></tr></thead>
            <tbody><tr v-for="g in data[side].groups" :key="g.category"><td><i class="class-dot" :class="g.category" />{{ labels[g.category] }}</td><td>{{ fmt(g.count) }}<small>{{ percent(g.count_pct) }}</small></td><td>{{ fmt(g.time_us) }}</td><td>{{ percent(g.time_pct) }}</td></tr></tbody>
          </table></div>
        </article>
      </div>
      <div class="counter-totals">
        <p>NPU aicore_time 合计：<strong>{{ fmt(data.npu.aicore_time_us) }} μs</strong><small>有效 {{ data.npu.aicore_time_measured_count }} / {{ data.npu.total_count }} 个 kernel</small></p>
        <p>NPU aiv_vec_time 合计：<strong>{{ fmt(data.npu.aiv_vec_time_us) }} μs</strong><small>有效 {{ data.npu.aiv_vec_time_measured_count }} / {{ data.npu.total_count }} 个 kernel</small></p>
      </div>
      <p class="muted small">占比分母为各端所选范围的全部计算 kernel。GPU 按已映射的 NPU 类别归组并去重；跨 Cube/Vector 的整组关联记为混合，涉及未知类别记为未知，未对齐 kernel 单列。GPU 内存活动 {{ data.gpu_reference.excluded_memory_count }} 个 / {{ fmt(data.gpu_reference.excluded_memory_time_us) }} μs 未计入此表。</p>
      <p class="muted small">GPU 类别仅用于工作负载参考，不代表 GPU 硬件单元或利用率。计数器缺失时合计仅包含有效值；aicore_time 与 aiv_vec_time 不相加为 kernel 总耗时。</p>
    </template>
  </section>
</template>

<style scoped>
.core-breakdown {margin-top:18px;}
.core-columns {display:grid;grid-template-columns:1fr 1fr;gap:24px;}
.core-columns article {min-width:0;}
.core-columns h3 {margin-bottom:4px;}
.class-bar {height:10px;border-radius:6px;overflow:hidden;display:flex;background:#edf1ef;margin:14px 0;}
.class-dot {display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;}
.Cube {background:#167d78;}.Vector {background:#7489ba;}.Mixed {background:#b68543;}.Unknown {background:#8c9198;}.Unmapped {background:#d4d8db;}
.counter-totals {display:flex;flex-wrap:wrap;gap:12px 36px;margin:16px 0;border-top:1px solid #e4ebe5;padding-top:10px;}
.counter-totals small {display:block;color:#758583;}
@media(max-width:1100px){.core-columns{grid-template-columns:1fr;gap:12px;}}
</style>
