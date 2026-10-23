'use strict';
const $=id=>document.getElementById(id);
const state={job:null,kind:'hardware',hardware:[],platform:[],offset:0,search:'',inventory:null,editing:null,poll:null};
const fmt=(v,n=2)=>v==null?'—':Number(v).toLocaleString('zh-CN',{maximumFractionDigits:n});
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n;};
function message(text){$('message').textContent=text;$('message').hidden=!text;}
async function api(path,options={}){const response=await fetch(path,{...options,headers:{...(options.body&&typeof options.body==='string'?{'Content-Type':'application/json'}:{}),...options.headers}});let data;try{data=await response.json();}catch{data={detail:response.statusText};}if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail||data));return data;}
const post=(path,data={})=>api(path,{method:'POST',body:JSON.stringify(data)});
function bind(id,handler){$(id).addEventListener('click',async()=>{message('');try{await handler();}catch(e){message(e.message);}});}
function option(select,value,text){const n=node('option',text);n.value=value;select.append(n);}
function clean(item){const {id,revision,created,...payload}=item;return payload;}
async function catalogs(){[state.hardware,state.platform]=await Promise.all(['hardware','platform'].map(k=>api('/api/catalog/'+k)));for(const id of ['source-platform','target-platform']){const s=$(id),old=s.value;s.replaceChildren();for(const p of state.platform)option(s,p.id,p.name+' · v'+p.revision);if(state.platform.some(p=>p.id===old))s.value=old;else{const p=state.platform.find(p=>id==='source-platform'?p.name.includes('L20'):p.name.includes('32'));if(p)s.value=p.id;}}}
function requireJob(){if(!state.job)throw Error('请先新建任务');return state.job.id;}
async function jobs(){const rows=await api('/api/jobs');$('jobs').replaceChildren();for(const r of rows){const b=node('button',r.name,'job'+(state.job?.id===r.id?' selected':''));b.append(node('small',r.status+' · '+r.created.slice(0,10)));b.onclick=()=>selectJob(r.id).catch(e=>message(e.message));$('jobs').append(b);}}
function restoreRequest(r){if(!r)return;$('side').value=r.scope.side;$('workload').value=r.workload;$('source-platform').value=r.source_platform_id;$('target-platform').value=r.target_platform_id;$('all-steps').checked=r.scope.all_steps;$('source-mode').value=r.assumptions.source_fp32_mode;$('target-mode').value=r.assumptions.target_fp32_mode;$('precision-confirm').checked=r.assumptions.precision_equivalence_confirmed;$('allow-estimates').checked=r.assumptions.allow_estimated_specs;$('serial-step').checked=r.assumptions.fixed_gap_serial_step;$('calibrations').value=JSON.stringify(r.calibrations,null,2);const {source_fp32_mode,target_fp32_mode,precision_equivalence_confirmed,allow_estimated_specs,fixed_gap_serial_step,...eff}=r.assumptions;$('assumptions').value=JSON.stringify(eff,null,2);}
async function selectJob(key){
  if(state.formBusy)return;
  rememberForm();
  if(state.job&&state.job.id!==key){$('trace-file').value='';$('detail-file').value='';}
  clearTimeout(state.poll);const epoch=(state.selectEpoch||0)+1;state.selectEpoch=epoch;
  const job=await api('/api/jobs/'+key);if(epoch!==state.selectEpoch)return;state.job=job;
  $('job-title').textContent=job.name;$('job-name').value=job.name;$('job-status').textContent=job.status;
  $('results').hidden=job.status!=='completed';state.inventory=null;
  for(const [id,label] of [['pid','唯一PID自动选择'],['step','请选择Step'],['device','唯一设备自动选择']]){const s=$(id);s.replaceChildren();option(s,'',label);}
  $('inventory').textContent='尚未预检查。选择文件后点击主按钮会自动上传；已完成任务须复用输入后修改。';
  $('precheck').checked=true;for(const id of ['pid','step','device'])$('manual-'+id).value='';
  $('all-steps').checked=false;restoreRequest(job.config?.request);renderEmpiricalSettings(job.config?.request?.empirical);
  if(job.config?.request?.scope)for(const id of ['pid','step','device'])$('manual-'+id).value=job.config.request.scope[id]||'';
  if(job.status==='ready'||job.status==='completed'){
    try{await loadInventory();}catch(e){if(job.status==='ready')message(e.message);}
    if(epoch!==state.selectEpoch)return;
    const scope=job.config?.request.scope;
    if(scope&&!state.inventory){for(const [id,value] of [['pid',scope.pid],['step',scope.step],['device',scope.device]]){if(value){option($(id),value,value+'（任务快照）');$(id).value=value;}}$('inventory').textContent='范围来自本次任务快照；该任务未保存独立预检查结果。';}
    if(scope&&state.inventory){$('pid').value=scope.pid||$('pid').value;fillSteps();
      const step=Array.from($('step').options).find(o=>o.value===scope.step||o.value==='ProfilerStep#'+scope.step);
      if(step)$('step').value=step.value;$('device').value=scope.device||$('device').value;}
  }
  if(job.status!=='completed')restoreForm(state.formDrafts?.[key]);
  const editable=['draft','ready','failed'].includes(job.status), busy=['running','inspecting','uploading'].includes(job.status);
  for(const id of ['job-name','side','source-platform','target-platform','trace-file','detail-file','pid','step','device','all-steps','workload','source-mode','target-mode','precision-confirm','allow-estimates','serial-step','assumptions','calibrations'])$(id).disabled=!editable;
  $('empirical-preview').disabled=job.status!=='completed';
  $('empirical-config').querySelectorAll('input,select').forEach(x=>x.disabled=busy);
  history.replaceState(null,'','?job='+key);await jobs();
  workflowControls();
  if(job.status==='completed')await results();if(busy)schedulePoll();if(job.error)message(job.error);
}
function schedulePoll(){
  clearTimeout(state.poll);const key=requireJob();
  state.poll=setTimeout(async()=>{try{if(state.job?.id!==key)return;const j=await api('/api/jobs/'+key);if(state.job?.id!==key)return;
    $('job-status').textContent=j.status;state.job=j;if(['running','inspecting','uploading'].includes(j.status))schedulePoll();else{await selectJob(j.id);if(state.job?.id===key&&j.status==='ready')message('预检查已完成，请确认范围后生成分析。');else if(state.job?.id===key&&j.status==='completed')message('分析已完成。');}
  }catch(e){message(e.message);}},1500);
}
async function newJob(){if(state.formBusy)return;const j=await post('/api/jobs',{name:$('job-name').value});await selectJob(j.id);}
async function loadInventory(){const key=requireJob(),epoch=state.selectEpoch;const data=await api('/api/jobs/'+key+'/inventory');if(state.job?.id!==key||epoch!==state.selectEpoch)return;state.inventory=data;$('inventory').textContent=`${fmt(data.trace_events,0)} 个trace事件，${fmt(data.detail_rows,0)} 行NPU明细。${data.warnings.join(' ')}`;const p=$('pid');p.replaceChildren();option(p,'','唯一PID自动选择');data.cpu_pids.forEach(x=>option(p,x.pid,x.pid+' · '+x.count+' 个CPU op'));if(data.cpu_pids.length===1)p.value=data.cpu_pids[0].pid;fillSteps();const d=$('device');d.replaceChildren();option(d,'','唯一设备自动选择');const dev=data.side==='npu'&&data.detail_devices?.length?data.detail_devices:data.devices;dev.forEach(x=>option(d,x,x));if(dev.length===1)d.value=dev[0];}
function fillSteps(){const s=$('step');s.replaceChildren();option(s,'','请选择Step');for(const x of state.inventory?.steps||[]){if(!x.pid||!$('pid').value||x.pid===$('pid').value)option(s,x.name,x.name+(x.duration_us?' · '+fmt(x.duration_us)+' μs':''));}}
$('pid').addEventListener('change',fillSteps);
bind('new-job',newJob);bind('refresh-jobs',jobs);
async function uploadLimits(){const m=await api('/api/meta');$('upload-limits').textContent=`单文件上限 ${fmt(m.upload_limit_mb,0)} MiB；解压/CSV上限 ${fmt(m.expanded_limit_mb,0)} MiB；分析内存软限制 ${fmt(m.memory_limit_mb,0)} MiB。支持 .json.gz，解压后仍须符合解析限制。`;return m;}
bind('rename',async()=>{const name=prompt('任务名称',state.job?.name);if(name){await api('/api/jobs/'+requireJob(),{method:'PATCH',body:JSON.stringify({name})});await selectJob(requireJob());}});
bind('clone',async()=>{const r=state.job?.config?.request;const j=await post('/api/jobs/'+requireJob()+'/clone');await selectJob(j.id);restoreRequest(r);renderEmpiricalSettings(r?.empirical);seedScope(r?.scope);rememberForm();workflowControls();message('已复制输入和配置。可预检查范围，或关闭预检查后确认已知范围直接分析。');});
bind('cancel',async()=>{await post('/api/jobs/'+requireJob()+'/cancel');await selectJob(requireJob());});
async function results(){
  const key=requireJob(),r=await api('/api/jobs/'+key+'/summary');
  if(state.job?.id!==key)return;
  state.summary=r;$('results').hidden=false;const p=r.prediction,s=r.profile_summary;
  updateResultHeadline(r);renderEmpirical(r.empirical);
  renderAssessment(r);
  $('result-scope').textContent=`${r.config.source_hardware.name} v${r.config.source_hardware.revision} → ${r.config.target_hardware.name} v${r.config.target_hardware.revision}；${r.config.request.scope.step||'全采集范围'}；源设备总时间 ${fmt(p.source_total_us)} μs。经验整体倍率与精细子集倍率分开展示，大于1表示目标更快。`;
  $('csv-link').href=`/api/jobs/${key}/download/operators.csv`;$('json-link').href=`/api/jobs/${key}/download/result.json`;
  $('class-title').textContent=s.classification_basis;$('breakdown').replaceChildren();
  for(const g of s.compute_breakdown){
    const row=node('div',null,'breakdown-row'),meter=node('meter');meter.max=100;meter.value=g.time_pct||0;meter.setAttribute('aria-label',g.category+'时间占比');
    row.append(node('span',g.category),meter,node('span',fmt(g.time_pct)+'% / '+fmt(g.time_us)+' μs'));$('breakdown').append(row);
  }
  renderDiagnostics(r);
  $('warnings').replaceChildren(...r.warnings.map(x=>node('li',x)));$('risks').replaceChildren();
  for(const risk of r.risks){const n=node('div',null,'risk');n.append(node('strong',risk.op+' · '+fmt(risk.source_us)+' μs'),node('span',risk.reason));$('risks').append(n);}
  $('snapshot').textContent=JSON.stringify({config:r.config,input_sha256:r.input_sha256,audit:r.audit,model_version:r.model_version},null,2);
  state.offset=0;await groups();
}
async function groups(){const data=await api(`/api/jobs/${requireJob()}/groups?offset=${state.offset}&limit=50&search=${encodeURIComponent(state.search)}`);$('groups').replaceChildren();for(const g of data.items){const tr=node('tr'),title=node('td');title.append(node('strong',g.op),node('small',g.shape||'Shape未知'),node('small',g.dtype||'dtype未知'));tr.append(title);for(const value of [g.class_name,g.calls+' / '+g.kernel_count,fmt(g.source_us),fmt(g.coverage_pct)+'%',fmt(g.predicted_us)])tr.append(node('td',value));tr.append(node('td',fmt(g.ratio),g.ratio==null?'':g.ratio>=2?'fast':g.ratio>=1?'good':'slow'));const td=node('td'),b=node('button','详情','secondary');b.onclick=async()=>{try{const d=await api(`/api/jobs/${requireJob()}/units?signature=${g.signature}`);$('detail-content').textContent=JSON.stringify({signature:g.signature,matched_calls:d.total,displayed_calls:d.items.length,group:g,units:d.items},null,2);$('detail-dialog').showModal();}catch(e){message(e.message);}};td.append(b);tr.append(td);$('groups').append(tr);}$('page-info').textContent=`${data.total} 组 · ${Math.floor(state.offset/50)+1} 页`;$('prev').disabled=state.offset===0;$('next').disabled=state.offset+50>=data.total;}
bind('search-button',async()=>{state.search=$('search').value;state.offset=0;await groups();});bind('prev',async()=>{state.offset=Math.max(0,state.offset-50);await groups();});bind('next',async()=>{state.offset+=50;await groups();});
function showResultTab(tab){document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x.dataset.tab===tab));for(const t of ['types','empirical','overview','evidence'])$('tab-'+t).hidden=t!==tab;}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>showResultTab(b.dataset.tab));
bind('catalog-new',()=>editCatalog(null));bind('editor-copy',()=>{state.editing=null;$('editor-title').textContent='复制为新条目';message('请修改名称后保存');});bind('editor-close',()=>$('editor').close());bind('detail-close',()=>$('detail-dialog').close());
for(const page of ['analysis','hardware','platform'])bind('nav-'+page,async()=>{document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.id==='nav-'+page));$('analysis-page').hidden=page!=='analysis';$('catalog-page').hidden=page==='analysis';if(page!=='analysis'){state.kind=page;await catalogs();renderCatalog();}});
$('assumptions').value=JSON.stringify({source_compute_efficiency:.6,target_compute_efficiency:.6,source_memory_efficiency:.7,target_memory_efficiency:.7,source_launch_us:3,target_launch_us:3},null,2);
initEmpiricalActions();
initWorkflow();
(async()=>{try{const meta=await uploadLimits();state.empiricalDefaults=meta.empirical_defaults;state.engineVersion=meta.version;renderEmpiricalSettings();await catalogs();await jobs();const id=new URLSearchParams(location.search).get('job');if(id)await selectJob(id);workflowControls();}catch(e){message(e.message);}})();
