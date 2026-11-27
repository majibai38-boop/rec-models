<script setup>
import {computed} from 'vue'
import {reportDownloads} from './downloads.js'
const props=defineProps({job:{type:Object,required:true}})
const files=computed(()=>reportDownloads(props.job))
const artifact=f=>'/api/jobs/'+props.job.id+'/artifacts/'+encodeURIComponent(f)
</script>
<template>
  <div class="result-meta"><slot/>
    <a v-if="files.excel" class="button primary" :href="artifact(files.excel)">↓ 下载完整 Excel</a>
    <span v-if="job.summary?.review_export_error" class="notice">{{ job.summary.review_export_error }}</span>
    <span v-if="job.summary?.aggregate_export_error" class="notice">{{ job.summary.aggregate_export_error }}</span>
    <details class="downloads"><summary>原始输入</summary><p class="muted small">上传时的完整文件，不按 Step 筛选。</p>
      <a v-for="i in files.inputs" :key="i.role" :href="'/api/jobs/'+job.id+'/inputs/'+i.role">{{ i.label }} · {{ i.name }}</a>
      <p v-if="!files.inputs.length" class="muted">暂无上传文件。</p>
    </details>
    <details class="downloads"><summary>更多输出</summary>
      <a v-for="[file,label] in files.more" :key="file" :href="artifact(file)">{{ label }} · {{ file }}</a>
      <p v-if="!files.more.length" class="muted">此报告没有其他可下载输出。</p>
    </details>
  </div>
</template>
