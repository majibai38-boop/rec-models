const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const app=fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8');
const management=fs.readFileSync(path.join(__dirname,'../static/management.js'),'utf8');

function setup(){
  const elements=new Map(),handlers={},calls=[];
  function element(tag='div'){
    let value='';const n={tagName:tag.toUpperCase(),children:[],dataset:{},checked:false,disabled:false,hidden:false,textContent:'',
      get value(){return value;},set value(v){value=n.tagName==='SELECT'&&!n.children.some(o=>o.value===v)?'':v;},
      append(...children){n.children.push(...children);},replaceChildren(...children){n.children=children;value='';},
      setAttribute(k,v){n[k]=v;},addEventListener(name,fn){handlers[(n.id||tag)+':'+name]=fn;},
      querySelectorAll(){return[];},showModal(){n.open=true;}};return n;
  }
  const $=id=>{if(!elements.has(id)){const n=element(id.endsWith('-platform')?'select':'div');n.id=id;elements.set(id,n);}return elements.get(id);};
  const context=vm.createContext({URLSearchParams,document:{getElementById:$,createElement:element},
    fetch:async()=>{throw Error('Unexpected fetch');},clearTimeout(){},history:{replaceState(){}},location:{pathname:'/'},
    renderEmpiricalSettings(){},restoreForm(){},workflowControls(){},selectJob:async()=>{},
  });
  vm.runInContext(app.slice(0,app.indexOf('function restoreRequest')),context);
  context.api=async(url,options)=>{calls.push([url,options]);return[];};
  // Function declarations can be rebound in this isolated VM without changing browser state.
  const run=code=>vm.runInContext(code,context);
  run('api = globalThis.api');
  const state=run('state');
  vm.runInContext(management,context);
  return {$,context,state,handlers,calls,run};
}

test('clean removes lifecycle metadata for strict catalog editing/import',()=>{
  const f=setup();const data=f.run('clean({id:"x",revision:2,created:"now",deleted_at:null,name:"device"})');
  assert.deepEqual(JSON.parse(JSON.stringify(data)),{name:'device'});
});
test('removed selected platform is not silently replaced on an existing task',async()=>{
  const f=setup();f.state.job={id:'job'};f.context.api=async url=>url.endsWith('platform')?[{id:'l20',name:'L20',revision:1}]:[];
  f.$('source-platform').append({value:'removed'});f.$('source-platform').value='removed';
  await f.run('catalogs()');assert.equal(f.$('source-platform').value,'');
});
test('deleted historical platform stays visible as snapshot',async()=>{
  const f=setup();f.state.job={id:'job',config:{source_platform:{id:'removed',name:'Old hardware',revision:3}}};
  f.$('source-platform').append({value:'removed'});f.$('source-platform').value='removed';
  await f.run('catalogs()');assert.equal(f.$('source-platform').value,'removed');
  assert.match(f.$('source-platform').children[1].textContent,/v3（已删除，仅历史快照）/);
});
test('task pagination displays fifty rows and keeps next-page sentinel',async()=>{
  const f=setup();f.context.api=async()=>Array.from({length:51},(_,i)=>({id:String(i),name:'task'+i,status:'draft',created:'2026-10-26'}));
  await f.run('jobs()');assert.equal(f.$('jobs').children.length,50);assert.equal(f.$('job-next').disabled,false);
});
test('completed job labels retain saved revision even when active platform was upgraded',async()=>{
  const f=setup();f.state.job={id:'job',status:'completed',config:{source_platform:{id:'p',name:'Original',revision:1}}};
  f.context.api=async url=>url.endsWith('platform')?[{id:'p',name:'Changed',revision:2}]:[];
  await f.run('catalogs()');assert.equal(f.$('source-platform').children[1].textContent,'Original · v1（任务快照）');
});
test('trash rows provide restore instead of opening a deleted task',async()=>{
  const f=setup();f.$('jobs-trash').checked=true;
  f.context.api=async url=>{f.calls.push(url);return url.includes('/restore')?{}:[{id:'x',name:'Recycled',deleted_at:'2026-10-26'}];};
  await f.run('jobs()');const row=f.$('jobs').children[0];assert.equal(row.tagName,'DIV');
  const button=row.children[2];assert.equal(button.textContent,'恢复');await button.onclick();
  assert.ok(f.calls.includes('/api/jobs/x/restore'));
});
test('page confirmation is cancel-first and Escape never confirms',async()=>{
  const f=setup();const promise=f.run('confirmRecycle("保留数据")');
  assert.equal(f.$('recycle-confirm').returnValue,'cancel');assert.equal(f.$('recycle-description').textContent,'保留数据');
  f.handlers['recycle-confirm:close']();assert.equal(await promise,false);
});
test('catalog recycling sends current version and explicit confirmation',async()=>{
  const f=setup();f.context.confirmRecycle=async()=>true;f.context.catalogs=async()=>{};f.context.loadCatalogPage=async()=>{};
  await f.run('recycleCatalog("hardware",{id:"x",name:"Chip",revision:7})');
  assert.equal(f.calls[0][0],'/api/catalog/hardware/x?confirm=true&expected_revision=7');assert.equal(f.calls[0][1].method,'DELETE');
});
