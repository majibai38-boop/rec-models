<script setup>
import { onMounted, onBeforeUnmount, watch, ref } from 'vue'
import { init, use } from 'echarts/core'
import { BarChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, MarkLineComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
use([BarChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, CanvasRenderer])
const props = defineProps({ option: Object })
const emit = defineEmits(['select'])
const el = ref(null)
let chart, observer
onMounted(() => { chart = init(el.value); chart.setOption(props.option); chart.on('click', p => emit('select', p)); observer = new ResizeObserver(() => chart.resize()); observer.observe(el.value) })
watch(() => props.option, v => chart?.setOption(v, true), { deep: true })
onBeforeUnmount(() => { observer?.disconnect(); chart?.dispose() })
</script>
<template><div ref="el" class="chart" role="img" aria-label="分析图表，数据同时显示在下方表格中"></div></template>
