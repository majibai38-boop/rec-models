import { test } from 'node:test'
import assert from 'node:assert/strict'
import { exportFiles, visibleDownloads, inputDownloads } from '../src/downloads.js'

test('more outputs contains only six requested exports, not internal evidence',()=>{
  const files=exportFiles.map(([name])=>name)
  assert.equal(files.length,6)
  assert.deepEqual(visibleDownloads([...files,'gpu_kernel_mapping.csv','coverage_diagnostics.json','npu_gpu_op_statistic.xlsx']).map(([n])=>n),files)
  assert.deepEqual(visibleDownloads(['gpu_kernel_mapping.csv']),[])
  assert.deepEqual(visibleDownloads(undefined),[])
})

test('original inputs shows only uploaded roles with original filenames',()=>{
  assert.deepEqual(inputDownloads(undefined),[])
  const items=inputDownloads({gpu_trace:{name:'model Step1+2.json'},npu_kernel_details:{name:''},private:{name:'secret'}})
  assert.deepEqual(items.map(i=>i.role),['gpu_trace','npu_kernel_details'])
  assert.equal(items[0].name,'model Step1+2.json')
  assert.equal(items[1].name,'kernel_details.csv')
})
