'use strict';
// Upload is part of the primary action. Precheck is optional, scope validation is not.
function selectedFiles(){return [[$('trace-file'),'trace'],[$('detail-file'),'detail']].filter(([input])=>input.files.length).map(([input,role])=>({input,file:input.files[0],name:role==='detail'?'kernel_details.csv':input.files[0].name.endsWith('.gz')?'trace.json.gz':'trace.json'}));}
function needsPrecheck(){return $('precheck').checked&&(!state.inventory||state.inventory.side!==$('side').value||selectedFiles().length>0);}
function captureForm(){
  const values={};$('job-form').querySelectorAll('input:not([type=file]),select,textarea').forEach(x=>{if(x.id)values[x.id]=x.type==='checkbox'?x.checked:x.value;});return values;
}
function restoreForm(values){
  if(!values)return;for(const [id,value]of Object.entries(values)){
    const input=$(id);if(!input)continue;
    if(input.type==='checkbox')input.checked=value;
    else if(input.tagName==='SELECT'){
      const match=Array.from(input.options).find(o=>o.value===value||(id==='step'&&o.value==='ProfilerStep#'+value));
      if(match)input.value=match.value;
      // A changed inventory may invalidate a previously chosen range; never invent a choice.
    }else input.value=value;
  }
}
function rememberForm(){if(state.job&&state.job.status!=='completed'){state.formDrafts??={};state.formDrafts[state.job.id]=captureForm();}}
function seedScope(scope){
  if(!scope)return;
  for(const id of ['pid','step','device']){$('manual-'+id).value=scope[id]||'';const select=$(id);if(scope[id]&&!Array.from(select.options).some(o=>o.value===scope[id]))option(select,scope[id],scope[id]+'（原范围，待核验）');select.value=scope[id]||'';}
  $('all-steps').checked=Boolean(scope.all_steps);
}
function workflowControls(){
  const busy=state.formBusy||['running','inspecting','uploading'].includes(state.job?.status),editable=!state.job||['draft','ready','failed'].includes(state.job.status);
  const checked=$('precheck').checked;
  $('checked-scope').hidden=!checked;$('manual-scope').hidden=checked;$('inventory').hidden=!checked;
  for(const id of ['precheck','manual-pid','manual-step','manual-device','all-steps'])$(id).disabled=busy||!editable;
  for(const id of ['pid','step','device'])$(id).disabled=busy||!editable||Boolean(selectedFiles().length)||!state.inventory;
  $('step').disabled||=$('all-steps').checked;$('manual-step').disabled||=$('all-steps').checked;
  $('run').disabled=Boolean(busy||!editable||!state.empiricalDefaults);
  $('run').textContent=state.formBusy?'正在提交…':state.job?.status==='inspecting'?'正在预检查…':state.job?.status==='running'?'正在分析…':needsPrecheck()?(selectedFiles().length?'上传并预检查范围':'预检查范围'):(selectedFiles().length?'上传并分析':'生成画像与预测');
  $('workflow-help').textContent=checked?'文件将在点击主按钮时自动上传。预检查完成后，确认PID、Step和设备，再点击“生成画像与预测”。':'跳过独立预检查，上传后直接分析。请填写Step或明确勾选全范围；PID/设备留空只允许唯一候选，不会默认选择最繁忙项。';
  if(checked&&selectedFiles().length)$('inventory').textContent='所选文件尚未上传；点击主按钮后会重新预检查，旧范围列表不能用于新输入。';
  if(busy)$('job-form').querySelectorAll('input,select,textarea').forEach(x=>x.disabled=true);
  // A browser upload cannot be cancelled by the worker's cancel action.
  $('cancel').disabled=Boolean(state.formBusy||!['running','inspecting'].includes(state.job?.status));
  $('new-job').disabled=Boolean(state.formBusy);$('clone').disabled=Boolean(busy||!state.job);$('rename').disabled=Boolean(state.formBusy||!state.job);
  $('delete-job').disabled=Boolean(busy||!state.job);
  $('jobs').querySelectorAll('button').forEach(x=>x.disabled=Boolean(state.formBusy));
}
async function uploadSelected(files,key){
  for(const {input,file,name}of files){
    message('正在上传 '+file.name+'，完成后自动继续…');
    await api(`/api/jobs/${key}/inputs/${name}`,{method:'PUT',body:file,headers:{'Content-Type':'application/octet-stream'}});
    input.value='';state.inventory=null;state.job.status='draft';
  }
}
function buildAnalysisRequest(){
  const src=state.platform.find(p=>p.id===$('source-platform').value),dst=state.platform.find(p=>p.id===$('target-platform').value);
  if(!src||!dst)throw Error('请选择源/目标平台');
  const prefix=$('precheck').checked?'':'manual-',all=$('all-steps').checked;
  const scope={side:$('side').value,pid:$(prefix+'pid').value.trim(),step:all?'':$(prefix+'step').value.trim(),device:$(prefix+'device').value.trim(),all_steps:all};
  if(!scope.step&&!all)throw Error('请填写或选择Step，或明确勾选全采集范围；不会自动分析全部Step。');
  return {name:$('job-name').value,source_platform_id:src.id,source_platform_revision:src.revision,target_platform_id:dst.id,target_platform_revision:dst.revision,scope,workload:$('workload').value,
    assumptions:{...JSON.parse($('assumptions').value),source_fp32_mode:$('source-mode').value,target_fp32_mode:$('target-mode').value,precision_equivalence_confirmed:$('precision-confirm').checked,allow_estimated_specs:$('allow-estimates').checked,fixed_gap_serial_step:$('serial-step').checked},empirical:readEmpiricalSettings(),calibrations:JSON.parse($('calibrations').value)};
}
async function submitWorkflow(){
  if(state.formBusy)return;
  const inspect=needsPrecheck(),side=$('side').value,files=selectedFiles();
  const request=inspect?null:buildAnalysisRequest(); // Validate before uploading, not afterwards.
  if(!state.job&&!files.length)throw Error('请先选择输入文件。');
  state.formBusy=true;workflowControls();
  try{
    const limits=await uploadLimits();
    for(const {file}of files)if(file.size>limits.upload_limit_mb*1024**2)throw Error(`${file.name} 超过单文件 ${limits.upload_limit_mb} MiB 上限；可通过 --upload-limit-mb 调整后重启服务`);
    if(!state.job){state.job=await post('/api/jobs',{name:$('job-name').value});history.replaceState(null,'','?job='+state.job.id);}
    const key=requireJob();rememberForm();
    await uploadSelected(files,key);
    if(inspect){await post('/api/jobs/'+key+'/inspect',{side});message('正在预检查；完成后请确认范围，再生成分析。');}
    else {await post('/api/jobs/'+key+'/run',request);message('已提交分析。');}
  }finally{
    state.formBusy=false;
    // Preserve values and successful uploads even when another file or the run request fails.
    if(state.job)await selectJob(state.job.id);
    else{$('job-form').querySelectorAll('input,select,textarea').forEach(x=>x.disabled=false);workflowControls();}
  }
}
function initWorkflow(){
  bind('run',submitWorkflow);
  $('precheck').addEventListener('change',()=>{
    if(!$('precheck').checked){for(const id of ['pid','step','device'])if($(id).value)$('manual-'+id).value=$(id).value;}
    else if(state.inventory){for(const id of ['pid','step','device']){const value=$('manual-'+id).value;if(Array.from($(id).options).some(o=>o.value===value))$(id).value=value;}}
    workflowControls();
  });
  for(const id of ['trace-file','detail-file','side'])$(id).addEventListener('change',workflowControls);
  $('all-steps').addEventListener('change',()=>{if($('all-steps').checked){$('step').value='';$('manual-step').value='';}workflowControls();});
}
