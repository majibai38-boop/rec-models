import test from 'node:test'
import assert from 'node:assert/strict'
import {traceRange,viewerUrl,waitForViewer,sendTrace} from '../src/traceViewer.js'

test('range uses absolute microseconds, independent sides and whole CPU scope',()=>{
  const unit={npu_kernels:[{timestamp_us:2000000,time_us:10},{timestamp_us:2100000,time_us:5}],
    npu_owners:[{timestamp_us:1900000,time_us:20}],gpu_kernels:[{timestamp_us:100000,time_us:1}]}
  assert.equal(traceRange(unit,'npu').timeStart,1.99999)
  assert.equal(traceRange(unit,'gpu').timeStart,.09999)
  assert.ok(traceRange(unit,'npu',0,true).timeStart<1.9)
  assert.equal(traceRange({},'npu'),null)
  assert.match(viewerUrl('/perfetto/',traceRange(unit,'npu')),/visStart=1999990000/)
})
test('handshake rejects other windows/origins and cleans up',async()=>{
  const host=new EventTarget(), messages=[], target={postMessage:(...args)=>messages.push(args)}
  const controller=new AbortController()
  const ready=waitForViewer(target,'http://local',controller.signal,host,1000)
  let resolved=false;ready.then(()=>resolved=true)
  const message=(source,origin)=>{const e=new Event('message');Object.assign(e,{source,origin,data:'PONG'});host.dispatchEvent(e)}
  message({},'http://local');message(target,'http://other');await Promise.resolve();assert.equal(resolved,false)
  message(target,'http://local');await ready
  assert.equal(messages[0][1],'http://local')
  const aborted=waitForViewer(target,'http://local',controller.signal,host,1000)
  controller.abort();await assert.rejects(aborted,/取消/)
})
test('stream transfer and failed HTTP do not silently open bad traces',async()=>{
  const messages=[], target={postMessage:(...args)=>messages.push(args)}
  await sendTrace(target,'http://local',new Response('{}',{headers:{'content-length':'2'}}),'trace')
  assert.equal(messages[0][0].perfetto.keepApiOpen,true)
  assert.equal(messages[0][1],'http://local')
  assert.equal(messages[0][2].length,1)
  await assert.rejects(sendTrace(target,'http://local',new Response('{"detail":"expired"}',{status:409}),'trace'),/expired/)
})
