import test from 'node:test'
import assert from 'node:assert/strict'
import {tensorText,numel,ratioClass} from '../src/operatorDetailView.js'
import {cleanView,readLocation,viewQuery} from '../src/viewState.js'

test('tensor metadata retains output, argument positions and missing values',()=>{
  assert.equal(tensorText({}),'输入：未采集')
  assert.equal(tensorText({output_shapes:[[4]],output_dtypes:['float16']},'output'),'output[0]: [4] · float16')
  assert.match(tensorText({input_dtypes:['float32','scalar']}),/input\[1\]: 未知 shape · scalar/)
})
test('numel rejects unknown, symbolic, TensorList and scalar arguments',()=>{
  assert.equal(numel({input_shapes:[[2,4]],input_dtypes:['float32']},0),8)
  assert.equal(numel({input_shapes:[[]],input_dtypes:['scalar']},0),null)
  assert.equal(numel({input_shapes:[[]],input_dtypes:['float32']},0),1)
  assert.equal(numel({input_shapes:[[[2],[4]]],input_dtypes:['tensorlist']},0),null)
  assert.equal(numel({input_shapes:[[-1,4]],input_dtypes:['float32']},0),null)
  assert.equal(numel({input_shapes:[[2,4]]},0),null)
  assert.equal(ratioClass(1),'ratio-light');assert.equal(ratioClass(2),'ratio-dark');assert.equal(ratioClass(null),'')
})
test('detail deep link keeps group, page, tensor selection and unit focus',()=>{
  const id='a'.repeat(32), view=cleanView({tab:'operator',scope:{op_type:'Add',device_id:'0',core_type:'AI_CORE'},operator:{search:'float16',group:'abc',unit:'unit:2',page:3,unitPage:4,input:1,sort:'ratio'}})
  assert.deepEqual(readLocation('?'+viewQuery(id,view)),{job:id,view})
  assert.equal(cleanView({operator:{input:-1,page:-2,sort:'SQL'}}).operator.input,0)
})
