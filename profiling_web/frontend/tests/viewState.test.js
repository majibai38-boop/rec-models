import test from 'node:test'
import assert from 'node:assert/strict'
import {cleanView,readLocation,viewQuery,readSavedViews,csvCell,normalizeStep} from '../src/viewState.js'

test('URL roundtrip preserves filters, navigation and event focus without secrets',()=>{
  const id='a'.repeat(32), view=cleanView({tab:'calls',page:3,recordPage:2,scope:{op_type:'Slice',device_id:'0',core_type:'AI_CORE'},callFilter:{group_id:'group:a',related_type:'Slice'},search:'<script>',hotspot:true,focus:'call:npu:12',token:'secret'})
  assert.deepEqual(readLocation('?'+viewQuery(id,view)),{job:id,view})
  assert.equal('token' in view,false)
})
test('untrusted saved state is bounded and defaults safely',()=>{
  assert.equal(readLocation('?job=../../etc&view=%7B').job,'')
  assert.equal(cleanView({tab:'bad',sort:'SQL',page:-2,recordPage:NaN}).tab,'overview')
  assert.equal(cleanView({page:-2}).page,0)
  assert.equal(readSavedViews('null').length,0)
  assert.equal(readSavedViews('[{"name":"saved","view":null}]')[0].view.tab,'overview')
})
test('filtered CSV neutralizes formulas and control prefixes but keeps numbers',()=>{
  assert.equal(csvCell('=HYPERLINK("bad")'),'"\'=HYPERLINK(""bad"")"')
  assert.equal(csvCell('\t123'),'"\'\t123"')
  assert.equal(csvCell('-2.5'),'"-2.5"')
  assert.equal(csvCell('normal'),'"normal"')
})
test('standard profiler labels normalize without changing custom scope names',()=>{
  assert.equal(normalizeStep('ProfilerStep#15'),'15')
  assert.equal(normalizeStep(' custom#15 '),'custom#15')
})
