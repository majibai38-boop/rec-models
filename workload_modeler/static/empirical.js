'use strict';
const empiricalScenarios=[['conservative','保守'],['baseline','基准'],['optimistic','乐观']];
const empiricalRateLabels={hardware_factors:'规格倍率乘数',missing_spec_ratio:'缺少规格时',fused_ratio:'FA/HSTU/混合融合',short_ratio:'短kernel',transfer_ratio:'内存活动',communication_ratio:'通信活动',unknown_ratio:'未知类别',gap_ratio:'Step非设备忙碌时间'};

function renderEmpiricalSettings(values){
  const a={...state.empiricalDefaults,...values},root=$('empirical-config');root.replaceChildren();
  function field(key,title,value,options={}){
    const label=node('label',title),input=node(options.choices?'select':'input');input.id='emp-'+key;
    if(options.choices){for(const x of options.choices)option(input,x,x.toUpperCase());input.value=value;}
    else if(typeof value==='boolean'){input.type='checkbox';input.checked=value;label.className='check';}
    else {input.type='number';input.step='any';input.min=options.min??0;input.max=options.max??1000;input.value=value??'';input.required=!options.optional;if(options.optional)input.placeholder='留空使用规格比';}
    input.setAttribute('aria-label',title);label.append(input);return label;
  }
  root.append(field('enabled','启用整体经验估算（独立于精细预测覆盖率）',a.enabled));
  const grid=node('div',null,'grid');
  grid.append(field('cube_compute_fraction','非短矩阵时间中计算部分占比（0–1）',a.cube_compute_fraction,{max:1}),field('short_kernel_us','短kernel阈值（小于此值，μs；0关闭）',a.short_kernel_us,{max:10000}));
  for(const side of ['source','target'])grid.append(field(side+'_matrix_mode',(side==='source'?'源':'目标')+'矩阵规格档位（整体假设，非检测精度）',a[side+'_matrix_mode'],{choices:['fp16','bf16','fp32','tf32','hf32']}));
  grid.append(field('matrix_ratio','自定义矩阵基准倍率（源时间/目标时间）',a.matrix_ratio,{optional:true}),field('bandwidth_ratio','自定义带宽基准倍率（源时间/目标时间）',a.bandwidth_ratio,{optional:true}));root.append(grid);
  const detail=node('details'),table=node('table'),head=node('tr');detail.append(node('summary','类别倍率与保守 / 基准 / 乐观假设'));
  for(const label of ['类别 / 系数',...empiricalScenarios.map(x=>x[1])])head.append(node('th',label));const thead=node('thead');thead.append(head);table.append(thead);const body=node('tbody');
  for(const [key,label]of Object.entries(empiricalRateLabels)){
    const row=node('tr');row.append(node('td',label));for(const [s,title]of empiricalScenarios){const td=node('td');td.append(field(key+'-'+s,label+' · '+title,a[key][s]));row.append(td);}body.append(row);
  }
  table.append(body);const wrap=node('div',null,'table-wrap');wrap.append(table);detail.append(wrap);root.append(detail);
  root.append(node('p','各倍率须大于0，保守 ≤ 基准 ≤ 乐观。规格倍率乘数默认0.7 / 1 / 1.3；其他类别默认基准1×是中性假设。范围是敏感性场景，不是置信区间。','muted'));
  root.oninput=()=>{$('empirical-edit-status').textContent=state.job?.status==='completed'?'假设已修改，当前结果尚未更新；点击“预览整体估算”应用。':'新分析将保存这些经验假设。';};
  $('empirical-edit-status').textContent='已完成任务可直接预览，不需重传或重解析Trace；预览不会改写原结果。';
}
function readEmpiricalSettings(){
  const a={};for(const key of Object.keys(state.empiricalDefaults)){
    const get=suffix=>{const input=$('emp-'+suffix);if(!input.checkValidity())throw Error(input.getAttribute('aria-label')+'：请填写有效数值');return input.value===''?null:Number(input.value);};
    if(key==='enabled')a[key]=$('emp-enabled').checked;
    else if(key.endsWith('_mode'))a[key]=$('emp-'+key).value;
    else if(empiricalRateLabels[key])a[key]=Object.fromEntries(empiricalScenarios.map(([s])=>[s,get(key+'-'+s)]));
    else a[key]=get(key);
  }return a;
}
function empiricalTable(headers,rows){
  const table=node('table'),head=node('thead'),tr=node('tr');headers.forEach(x=>tr.append(node('th',x)));head.append(tr);table.append(head);
  const body=node('tbody');for(const row of rows){const r=node('tr');row.forEach(x=>r.append(node('td',x)));body.append(r);}table.append(body);const wrap=node('div',null,'table-wrap');wrap.append(table);return wrap;
}
function renderEmpirical(e){
  const root=$('empirical-output');root.replaceChildren();state.empiricalResult=e;$('empirical-export').disabled=!e;
  if(!e){root.append(node('p','暂无经验估算。'));return;}
  root.append(node('p',e.note||'本结果已随分析保存；下载完整JSON可查看全部经验假设与依据。','muted'));
  if(e.status!=='available'){root.append(node('p',e.conclusion,'note'));return;}
  root.append(node('p',e.conclusion,'note'));
  root.append(empiricalTable(['场景','源设备时间 ms','目标设备时间 ms','设备源/目标','设备耗时减少','目标Step ms','Step源/目标'],empiricalScenarios.map(([s,title])=>[title,fmt(e.source_total_us/1000,3),fmt(e.device[s].target_us/1000,3),fmt(e.device[s].ratio,4)+'×',fmt(e.device[s].time_reduction_pct)+'%',e.step?fmt(e.step.scenarios[s].target_us/1000,3):'—',e.step?fmt(e.step.scenarios[s].ratio,4)+'×':'—'])));
  if(e.step){root.append(node('p',`源Step ${fmt(e.step.source_us/1000,3)} ms，其中设备忙碌 ${fmt(e.step.source_busy_us/1000,3)} ms，非设备忙碌 ${fmt(e.step.source_gap_us/1000,3)} ms（${fmt(e.step.source_gap_us/e.step.source_us*100)}%）。间隙不是已确认的CPU开销。`,'note'),node('p',e.step.basis,'muted'));}
  else root.append(node('p',e.step_reason,'muted'));
  root.append(node('h3','设备时间拆分与折算依据'),node('p',`类别独立 / 缺规格倍率假设覆盖 ${fmt(e.direct_assumption_time_pct)}% 源设备时间；其余也依赖规格与实现效率假设。所有分组互斥，源耗时守恒。`,'muted'));
  root.append(empiricalTable(['类别','源时间 μs','占比','保守倍率','基准倍率','乐观倍率','目标基准 μs','依据'],e.buckets.map(b=>[b.label,fmt(b.source_us),fmt(b.source_pct)+'%',...empiricalScenarios.map(([s])=>fmt(b.ratios[s],3)+'×'),fmt(b.target_us.baseline),b.basis])));
  root.append(node('p',e.formula,'muted'));const details=node('details'),list=node('ul');details.open=true;details.append(node('summary','经验假设与限制'));e.notices.forEach(x=>list.append(node('li',x)));details.append(list);root.append(details);
}
function updateResultHeadline(r){
  const p=r.prediction,e=r.empirical;$('cards').replaceChildren();
  for(const [label,value]of [['设备整体 · 经验倍率',e?.device?fmt(e.device.baseline.ratio,3)+'×':e?.status==='disabled'?'已关闭':'暂无设备时间'],['整步 · 经验倍率',e?.step?fmt(e.step.scenarios.baseline.ratio,4)+'×':e?.status==='disabled'?'已关闭':'缺少Step依据'],['精细预测覆盖率',fmt(p.coverage_pct)+'%'],['精细预测 · 已覆盖子集',p.modelled_device_ratio==null?'尚不可量化':fmt(p.modelled_device_ratio)+'×']]){const c=node('div',null,'card');c.append(node('span',label),node('strong',value));$('cards').append(c);}
  $('verdict').textContent=e?.status==='available'?e.conclusion:r.assessment?.screening.conclusion||p.verdict;
}
function initEmpiricalActions(){
  bind('empirical-jump',()=>{$('empirical-settings').open=true;$('empirical-settings').scrollIntoView({behavior:'smooth',block:'start'});});
  bind('empirical-preview',async()=>{
    const key=requireJob(),epoch=state.selectEpoch;if(state.job.status!=='completed')throw Error('任务完成后才能预览。');
    const options=readEmpiricalSettings();$('empirical-preview').disabled=true;
    $('empirical-config').querySelectorAll('input,select').forEach(x=>x.disabled=true);
    try{const e=await post('/api/jobs/'+key+'/empirical-preview',options);if(state.job?.id!==key||epoch!==state.selectEpoch)return;
      state.summary.empirical=e;renderEmpirical(e);updateResultHeadline(state.summary);showResultTab('empirical');$('empirical-edit-status').textContent='预览已更新；原任务与精细预测未改写。';$('tab-empirical').scrollIntoView({behavior:'smooth',block:'start'});
    }finally{if(state.job?.id===key&&epoch===state.selectEpoch){$('empirical-preview').disabled=false;$('empirical-config').querySelectorAll('input,select').forEach(x=>x.disabled=false);}}
  });
  bind('empirical-export',()=>{
    const e=state.empiricalResult;if(!e)throw Error('暂无经验结果');
    const r=state.summary,data={...e,provenance:e.provenance||{job_id:requireJob(),input_sha256:r.input_sha256,config:r.config,engine_version:state.engineVersion}};
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),link=node('a');link.href=url;link.download='empirical-'+requireJob()+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
}
