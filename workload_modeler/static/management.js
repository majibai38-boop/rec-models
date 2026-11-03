'use strict';
// Recycling only hides entries; input files, results and revision snapshots stay intact.
function confirmRecycle(text){
  const dialog=$('recycle-confirm');
  if(dialog.open)return Promise.resolve(false);
  $('recycle-description').textContent=text;dialog.returnValue='cancel';
  return new Promise(resolve=>{dialog.addEventListener('close',()=>resolve(dialog.returnValue==='confirm'),{once:true});dialog.showModal();});
}
async function resetJobView(){
  clearTimeout(state.poll);state.selectEpoch=(state.selectEpoch||0)+1;
  state.job=null;state.inventory=null;state.summary=null;state.empiricalPreview=null;
  history.replaceState(null,'',location.pathname);
  $('trace-file').value='';$('detail-file').value='';$('results').hidden=true;
  $('job-title').textContent='新建分析';jobStatus('未选择');
  for(const [id,label]of [['pid','唯一PID自动选择'],['step','请选择Step'],['device','唯一设备自动选择']]){
    $(id).replaceChildren();option($(id),'',label);
  }
  $('job-form').querySelectorAll('input,select,textarea').forEach(x=>x.disabled=false);
  renderEmpiricalSettings();await catalogs();restoreForm(state.emptyForm);
  $('empirical-preview').disabled=true;$('inventory').textContent='尚未预检查；请选择输入文件。';
  workflowControls();
}
bind('delete-job',async()=>{
  if(state.formBusy||!state.job)return;
  const key=requireJob();
  if(!await confirmRecycle('将“'+state.job.name+'”移入回收站？\n输入文件、结果与历史快照保留，可恢复；不会释放磁盘空间。'))return;
  state.formBusy=true;workflowControls();
  try{
    await api('/api/jobs/'+key+'?confirm=true',{method:'DELETE'});
    await resetJobView();await jobs();message('任务已移入回收站。勾选左侧“回收站”即可恢复。');
  }finally{
    state.formBusy=false;
    if(state.job)await selectJob(state.job.id);
    else{$('job-form').querySelectorAll('input,select,textarea').forEach(x=>x.disabled=false);workflowControls();}
  }
});
async function findJobs(){state.jobSearch=$('job-search').value.trim();state.jobOffset=0;await jobs();}
bind('find-jobs',findJobs);
$('job-search').addEventListener('keydown',e=>{if(e.key==='Enter')$('find-jobs').click();});
$('jobs-trash').addEventListener('change',()=>findJobs().catch(e=>message(e.message)));
bind('job-prev',async()=>{state.jobOffset=Math.max(0,(state.jobOffset||0)-50);await jobs();});
bind('job-next',async()=>{state.jobOffset=(state.jobOffset||0)+50;await jobs();});

async function loadCatalogPage(){
  const epoch=state.catalogEpoch=(state.catalogEpoch||0)+1,kind=state.kind;
  const rows=await api('/api/catalog/'+kind+'?deleted='+$('catalog-trash').checked);
  if(epoch!==state.catalogEpoch||kind!==state.kind)return;
  state.catalogRows=rows;renderCatalog();
}
$('catalog-trash').addEventListener('change',()=>loadCatalogPage().catch(e=>message(e.message)));
$('catalog-search').addEventListener('input',()=>renderCatalog());
async function recycleCatalog(kind,item,restore=false){
  if(!restore&&!await confirmRecycle('将“'+item.name+'”移入回收站？\n历史版本和分析结果保留。芯片被有效平台引用时，需要先修改或删除对应平台。'))return;
  const path=`/api/catalog/${kind}/${item.id}`;
  if(restore)await post(path+'/restore?expected_revision='+item.revision);
  else await api(path+'?confirm=true&expected_revision='+item.revision,{method:'DELETE'});
  await catalogs();await loadCatalogPage();
  message(restore?'已恢复；仍保留原来的版本号和引用。':'已移入回收站，历史分析快照不受影响。可勾选“回收站”恢复。');
}
