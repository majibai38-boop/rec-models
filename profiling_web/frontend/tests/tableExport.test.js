import {test} from 'node:test'
import assert from 'node:assert/strict'
import {filteredCsv} from '../src/tableExport.js'
import {reportDownloads} from '../src/downloads.js'

test('both modes offer their own workbook, available files and original names',()=>{
  const pair=reportDownloads({mode:'npu_pair',summary:{reports:['npu_pair.xlsx','npu_pair.json']},inputs:{baseline_details:{name:'details.csv'},candidate_details:{name:'details.csv'}}})
  assert.equal(pair.excel,'npu_pair.xlsx');assert.equal(pair.more.length,1)
  assert.deepEqual(pair.inputs.map(i=>i.role),['baseline_details','candidate_details'])
  assert.equal(reportDownloads({summary:{reports:[]}}).excel,null)
  assert.deepEqual(reportDownloads({summary:{reports:['call_tables.xlsx','private.json']}}).more.map(x=>x[0]),['call_tables.xlsx'])
})
test('filtered CSV keeps the scope across pages, visible columns and formula escaping',async()=>{
  const urls=[]
  const csv=await filteredCsv(async url=>{urls.push(url);return {total:2,items:[{name:urls.length===1?'=formula':'last',private:1}]}
  },'/table?search=Add',{columns:['name']})
  assert.deepEqual(urls,['/table?search=Add&limit=200&offset=0','/table?search=Add&limit=200&offset=1'])
  assert.equal(csv,'\uFEFF"name"\r\n"\'=formula"\r\n"last"')
  await assert.rejects(filteredCsv(async()=>({total:3,items:[]}),'/table'),/不完整/)
})
