const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../static/workflow.js'),'utf8');

function setup(){
  const elements=new Map(),calls=[],handlers={};
  function $(id){
    if(!elements.has(id)){
      let value='';
      const e={id,tagName:'INPUT',type:'text',checked:false,disabled:false,hidden:false,files:[],options:[],
        get value(){return value;},set value(v){value=v;if(e.type==='file'&&v==='')e.files=[];},
        addEventListener:(event,fn)=>{handlers[id+':'+event]=fn;},
        querySelectorAll:selector=>id==='job-form'?[...elements.values()].filter(x=>x.tagName==='INPUT'&&(!selector.includes(':not')||x.type!=='file')):[]};
      elements.set(id,e);
    }return elements.get(id);
  }
  for(const id of ['trace-file','detail-file'])$(id).type='file';
  for(const id of ['precheck','all-steps','precision-confirm','allow-estimates','serial-step'])$(id).type='checkbox';
  $('precheck').checked=true;$('side').value='gpu';$('assumptions').value='{}';$('calibrations').value='[]';$('job-name').value='test';
  $('source-platform').value='s';$('target-platform').value='t';
  const state={job:{id:'test-job',status:'draft'},platform:[{id:'s',revision:1},{id:'t',revision:1}],empiricalDefaults:{},inventory:null};
  const context=vm.createContext({$,state,history:{replaceState(){}},
    option:(input,value)=>input.options.push({value}),bind:(id,fn)=>{handlers[id]=fn;},message:()=>{},
    requireJob:()=>state.job.id,readEmpiricalSettings:()=>({enabled:true}),
    uploadLimits:async()=>({upload_limit_mb:2}),
    api:async(url,opts)=>{calls.push(['upload',url]);return{};},
    post:async(url,data)=>{calls.push(['post',url,data]);return{id:'created',status:'draft'};},
    selectJob:async key=>{calls.push(['select',key]);},
  });
  vm.runInContext(source,context);
  return {context,$,state,calls,handlers,run:code=>vm.runInContext(code,context)};
}

test('precheck uploads once then inspects, without running analysis',async()=>{
  const f=setup();f.$('trace-file').files=[{name:'sample.json.gz',size:123}];
  await f.run('submitWorkflow()');
  assert.deepEqual(f.calls.map(x=>x.slice(0,2)),[['upload','/api/jobs/test-job/inputs/trace.json.gz'],['post','/api/jobs/test-job/inspect'],['select','test-job']]);
  assert.equal(f.$('trace-file').files.length,0);assert.equal(f.state.formBusy,false);
});
test('disabled precheck requires explicit scope before any upload',async()=>{
  const f=setup();f.$('precheck').checked=false;f.$('trace-file').files=[{name:'sample.json',size:123}];
  await assert.rejects(f.run('submitWorkflow()'),/Step/);assert.equal(f.calls.length,0);
});
test('disabled precheck uploads then runs with manual scope',async()=>{
  const f=setup();f.$('precheck').checked=false;f.$('manual-step').value='15';f.$('manual-pid').value='123';f.$('trace-file').files=[{name:'sample.json',size:123}];
  await f.run('submitWorkflow()');
  assert.equal(f.calls[1][1],'/api/jobs/test-job/run');assert.equal(f.calls[1][2].scope.step,'15');assert.equal(f.calls[1][2].scope.pid,'123');
  assert.equal(f.calls.some(x=>x[1].endsWith('/inspect')),false);
});
test('ready inventory runs without reupload or repeat inspection',async()=>{
  const f=setup();f.state.inventory={side:'gpu'};f.$('step').value='ProfilerStep#1';
  await f.run('submitWorkflow()');assert.equal(f.calls[0][1],'/api/jobs/test-job/run');assert.equal(f.calls.length,2);
});
test('new files invalidate ready inventory',()=>{
  const f=setup();f.state.inventory={side:'gpu'};assert.equal(f.run('needsPrecheck()'),false);
  f.$('detail-file').files=[{name:'kernel_details.csv',size:12}];assert.equal(f.run('needsPrecheck()'),true);
});
test('failed second upload keeps remaining file and never starts worker',async()=>{
  const f=setup();f.$('trace-file').files=[{name:'sample.json',size:123}];f.$('detail-file').files=[{name:'kernel_details.csv',size:123}];
  f.context.api=async url=>{f.calls.push(['upload',url]);if(url.endsWith('.csv'))throw Error('upload failed');};
  await assert.rejects(f.run('submitWorkflow()'),/upload failed/);
  assert.equal(f.$('trace-file').files.length,0);assert.equal(f.$('detail-file').files.length,1);
  assert.equal(f.calls.some(x=>x[0]==='post'),false);assert.equal(f.state.formBusy,false);
});
test('form busy prevents duplicate submission',async()=>{
  const f=setup();f.$('trace-file').files=[{name:'sample.json',size:123}];
  let release;f.context.uploadLimits=()=>new Promise(resolve=>release=resolve);
  const first=f.run('submitWorkflow()');await f.run('submitWorkflow()');release({upload_limit_mb:2});await first;
  assert.equal(f.calls.filter(x=>x[0]==='upload').length,1);
});
test('precheck toggle carries known ranges and all-range is explicit',()=>{
  const f=setup();f.run('initWorkflow()');f.$('pid').value='1';f.$('step').value='15';f.$('precheck').checked=false;
  f.handlers['precheck:change']();assert.equal(f.$('manual-step').value,'15');assert.equal(f.$('all-steps').checked,false);
  f.$('all-steps').checked=true;f.handlers['all-steps:change']();assert.equal(f.$('manual-step').value,'');assert.equal(f.$('manual-step').disabled,true);
});
test('draft restore retains assumptions and accepts canonical Step alias',()=>{
  const f=setup();f.$('step').tagName='SELECT';f.$('step').options=[{value:''},{value:'ProfilerStep#15'}];
  f.context.draft={'step':'15','manual-step':'15','precheck':false,'assumptions':'{"source_launch_us":5}'};
  f.run('restoreForm(draft)');assert.equal(f.$('step').value,'ProfilerStep#15');assert.equal(f.$('precheck').checked,false);assert.equal(f.$('assumptions').value,'{"source_launch_us":5}');
});
