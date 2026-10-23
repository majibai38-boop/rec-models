'use strict';
// Data is rendered using textContent and form controls, never interpreted as HTML.
const metricLabels = {
  matrix_fp16:'矩阵 FP16', matrix_bf16:'矩阵 BF16', matrix_fp32:'矩阵 FP32', matrix_tf32:'矩阵 TF32', matrix_hf32:'矩阵 HF32',
  vector_fp16:'Vector FP16', vector_bf16:'Vector BF16', vector_fp32:'Vector FP32', memory_bandwidth:'显存带宽',
  memory_capacity:'显存容量', l2_bandwidth:'L2 带宽', launch_latency:'启动延迟', cube_count:'Cube 核数', vector_count:'Vector 核数'
};
const platformFields = {
  card_count:['设备数量',true,1], cpu_model:['CPU 型号',false], cpu_count:['CPU 数量',true],
  cores_per_cpu:['每 CPU 核数',true], cpu_mhz:['CPU 频率 MHz',true], dimm_count:['内存条数',true],
  dimm_gb:['单条容量 GB',true], memory_rated_mts:['内存额定速率 MT/s',true],
  memory_configured_mts:['内存实际速率 MT/s',true], topology:['拓扑说明',false], software:['软件与版本',false]
};
let editorPayload = {}, editorMode = 'form', editorMetrics = {};

function formField(parent,key,title,value,{numeric=false,choices=null,multiline=false}={}) {
  const label=node('label',title), input=node(choices?'select':multiline?'textarea':'input');
  input.dataset.field=key;
  if (choices) for (const [v,t] of choices) option(input,v,t);
  else if(numeric){input.type='number';input.step='any';input.min='0';input.placeholder='未知请留空';}
  else if(multiline) input.rows=2;
  input.value=value??'';label.append(input);parent.append(label);return input;
}
function field(key){return $('editor-fields').querySelector(`[data-field="${key}"]`);}
function val(key,numeric=false){const input=field(key),v=input.value.trim();if(!numeric)return v;if(input.validity.badInput)throw Error('数字尚未填写完整：'+key);if(v==='')return null;const n=Number(v);if(!Number.isFinite(n))throw Error('请填写有限数字：'+key);return n;}
function checkFormPayload(payload){
  if(!payload||typeof payload!=='object'||Array.isArray(payload))throw Error('请输入单条JSON对象');
  if(state.kind==='hardware') {
    for(const [key,m] of Object.entries(payload.metrics||{})){
      if(!editorMetrics[key])throw Error('未知指标：'+key+'；请在高级JSON中核对');
      if(!m||m.unit!==editorMetrics[key])throw Error(key+' 单位必须为 '+editorMetrics[key]);
    }
  }
}
async function fillCatalogForm(payload){
  checkFormPayload(payload);editorPayload=structuredClone(payload);const root=$('editor-fields');root.replaceChildren();
  const common=node('div',null,'grid');root.append(common);formField(common,'name','名称',payload.name);
  if(state.kind==='hardware'){
    formField(common,'vendor','芯片厂商',payload.vendor||'Ascend',{choices:[['Ascend','Ascend'],['NVIDIA','NVIDIA'],['Other','Other']]});
    root.append(node('p','空值保持未知。每个有效数值须有证据类型及来源；矩阵稠密口径未确认时不用于预测。','note'));
    for(const [key,unit] of Object.entries(editorMetrics)){
      const m=payload.metrics?.[key]||{}, box=node('fieldset');box.className='metric-editor';
      box.append(node('legend',(metricLabels[key]||key)+' · '+unit));const grid=node('div',null,'grid');box.append(grid);root.append(box);
      formField(grid,key+'.value','数值',m.value,{numeric:true});
      formField(grid,key+'.evidence','证据类型',m.evidence||'unknown',{choices:[['unknown','未知'],['official','公开规格'],['measured','实测'],['estimated','估算'],['user','用户确认']]});
      formField(grid,key+'.source','来源 / 测试条件',m.source||'');formField(grid,key+'.note','备注',m.note||'');
      if(key.startsWith('matrix_'))formField(grid,key+'.dense','稠密算力口径',m.dense==null?'':String(m.dense),{choices:[['','未确认'],['true','已确认稠密'],['false','非稠密 / 稀疏']]});
    }
  }else{
    // Include old hardware revisions so editing a platform never silently upgrades it.
    const histories=await Promise.all(state.hardware.map(h=>api(`/api/catalog/hardware/${h.id}/history`)));
    const choices=histories.flat().map(h=>[h.id+'@'+h.revision,h.name+' · v'+h.revision]);
    const selected=payload.hardware_id?payload.hardware_id+'@'+payload.hardware_revision:'';
    formField(common,'hardware','芯片及固定版本',selected,{choices:[['','请选择芯片版本'],...choices]});
    for(const [k,[label,numeric,def]]of Object.entries(platformFields))formField(common,k,label,payload[k]??def??'',{numeric});
    root.append(node('p','芯片修改不会自动升级此引用。设备数量大于1可保存配置，但当前预测引擎只接受单设备平台。','note'));
  }
  formField(root,'note','条目备注',payload.note||'',{multiline:true});
}
function readCatalogForm(){
  const payload=structuredClone(editorPayload);payload.name=val('name');payload.note=val('note');
  if(state.kind==='hardware'){
    payload.vendor=val('vendor');payload.metrics={};
    for(const [key,unit]of Object.entries(editorMetrics)){
      const value=val(key+'.value',true),source=val(key+'.source'),note=val(key+'.note'),evidence=val(key+'.evidence');
      const old=editorPayload.metrics?.[key];
      const dense=key.startsWith('matrix_')?(val(key+'.dense')===''?null:val(key+'.dense')==='true'):(old?.dense??null);
      if(old||value!==null||source||note||evidence!=='unknown'||dense!==null)payload.metrics[key]={...old,value,unit,source,note,evidence,dense};
    }
  }else{
    const [id,revision]=val('hardware').split('@');if(!id)throw Error('请选择芯片版本');
    payload.hardware_id=id;payload.hardware_revision=Number(revision);
    for(const [k,[,numeric]]of Object.entries(platformFields))payload[k]=val(k,numeric);
  }
  return payload;
}
async function changeEditorMode(mode){
  if(mode===editorMode)return;
  if(mode==='json')$('editor-json').value=JSON.stringify(readCatalogForm(),null,2);
  else await fillCatalogForm(JSON.parse($('editor-json').value));
  editorMode=mode;$('editor-json').hidden=mode!=='json';$('editor-fields').hidden=mode!=='form';
}
async function editCatalog(item){
  state.editing=item||null;$('editor-error').textContent='';$('editor-import').value='';
  $('editor-title').textContent=(item?'修改':'新增')+(state.kind==='hardware'?'芯片规格':'平台配置');
  $('editor-help').textContent='修改会创建新版本；历史分析仍使用原快照。JSON导入仅预填当前条目，不批量覆盖规格库。';
  editorMetrics=(await api('/api/meta')).metric_units;editorMode='form';$('editor-json').hidden=true;$('editor-fields').hidden=false;
  await fillCatalogForm(item?clean(item):{name:'',note:'',...(state.kind==='hardware'?{vendor:'Ascend',metrics:{}}:{card_count:1})});
  $('editor').showModal();
}
function downloadCatalog(item){
  const url=URL.createObjectURL(new Blob([JSON.stringify(clean(item),null,2)],{type:'application/json'}));
  const a=node('a');a.href=url;a.download=state.kind+'-'+item.id+'-v'+item.revision+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function renderCatalog(){
  const kind=state.kind;$('catalog-title').textContent=kind==='hardware'?'芯片规格库':'平台配置库';$('catalog-list').replaceChildren();
  for(const item of state[kind]){
    const card=node('article');card.append(node('h3',item.name+' · v'+item.revision));const dl=node('dl');
    if(kind==='hardware')for(const [k,m]of Object.entries(item.metrics)){dl.append(node('dt',metricLabels[k]||k),node('dd',fmt(m.value)+' '+m.unit+' · '+m.evidence+(k.startsWith('matrix_')?' / 稠密：'+(m.dense===true?'是':m.dense===false?'否':'未确认'):'')));}
    else{const hw=state.hardware.find(h=>h.id===item.hardware_id);dl.append(node('dt','芯片固定版本'),node('dd',(hw?.name||item.hardware_id)+' · v'+item.hardware_revision));for(const [k,[title]]of Object.entries(platformFields))dl.append(node('dt',title),node('dd',item[k]??'未知'));}
    card.append(dl,node('p',item.note,'muted'));const tools=node('div',null,'toolbar');
    for(const [title,fn]of [['修改新版本',()=>editCatalog(item)],['版本历史',async()=>{$('detail-content').textContent=JSON.stringify(await api(`/api/catalog/${kind}/${item.id}/history`),null,2);$('detail-dialog').showModal();}],['导出JSON',()=>downloadCatalog(item)]]){
      const b=node('button',title,'secondary');b.onclick=()=>Promise.resolve().then(fn).catch(e=>message(e.message));tools.append(b);
    }card.append(tools);$('catalog-list').append(card);
  }
}
$('editor-form').onsubmit=async e=>{
  e.preventDefault();$('editor-save').disabled=true;
  try{
    const payload=editorMode==='json'?JSON.parse($('editor-json').value):readCatalogForm(),x=state.editing;
    if(x)await api(`/api/catalog/${state.kind}/${x.id}`,{method:'PUT',body:JSON.stringify({expected_revision:x.revision,payload})});
    else await post('/api/catalog/'+state.kind,payload);
    $('editor').close();await catalogs();renderCatalog();message('新版本已保存；平台需显式选择新芯片版本，历史结果不变。');
  }catch(e){$('editor-error').textContent=e.message;}finally{$('editor-save').disabled=false;}
};
bind('editor-form-mode',()=>changeEditorMode('form'));bind('editor-json-mode',()=>changeEditorMode('json'));
$('editor-import').onchange=async()=>{
  try{const f=$('editor-import').files[0];if(!f)return;if(f.size>1024**2)throw Error('单条JSON不得超过1MiB');
    const payload=clean(JSON.parse(await f.text()));$('editor-json').value=JSON.stringify(payload,null,2);
    await fillCatalogForm(payload);editorMode='form';$('editor-json').hidden=true;$('editor-fields').hidden=false;$('editor-error').textContent='已预填，尚未保存。修改现有条目将新增版本。';
  }catch(e){$('editor-error').textContent=e.message;}
};
