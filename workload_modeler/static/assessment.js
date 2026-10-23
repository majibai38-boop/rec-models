'use strict';
function renderAssessment(result){
  const root=$('assessment-layers');root.replaceChildren();
  const a=result.assessment;
  if(!a){root.append(node('p','旧结果尚无分层结论，请复用输入重新分析。','note'));return;}
  if(a.historical)root.append(node('p',a.note,'note historical-note'));
  const m=a.screening.metrics;
  const stages=[
    ['1 · 画像与亲和性初筛',a.screening.conclusion,`分类覆盖 ${fmt(m.classified_pct)}% · 未知 ${fmt(m.unknown_pct)}%`, '查看画像与建议','overview'],
    ['2 · 精细模型：已覆盖部分',a.partial.conclusion,`精细预测覆盖 ${fmt(a.partial.coverage_pct)}% · 未覆盖 ${fmt(a.partial.uncovered_source_us)} μs`,'查看未建模原因','evidence'],
    ['3 · 精细模型：严格整步预测',a.step.conclusion,a.step.prediction?`源/目标 ${fmt(a.step.prediction.ratio)}×`:'严格条件尚未满足，不影响独立的经验估算','查看整步条件','overview']
  ];
  for(const [title,text,metric,action,tab] of stages){
    const box=node('article',null,'assessment-card');box.append(node('h3',title),node('p',text),node('p',metric,'muted'));
    const button=node('button',action,'secondary');button.onclick=()=>{showResultTab(tab);$(tab==='overview'?'tab-overview':'tab-evidence').scrollIntoView({block:'start',behavior:'smooth'});};box.append(button);root.append(box);
  }
  $('screening-conclusion').textContent=a.screening.conclusion;
  $('screening-limit').textContent=a.screening.limitation;
  $('screening-signals').replaceChildren();
  for(const signal of a.screening.signals){const box=node('div',null,'risk');box.append(node('strong',signal.title),node('span',signal.detail));$('screening-signals').append(box);}
  $('screening-priorities').replaceChildren();
  for(const p of a.screening.priorities){
    const box=node('article',null,'diagnostic');box.append(node('h3',p.op),node('p',`${fmt(p.source_us)} μs · 全部设备时间的 ${fmt(p.source_pct)}% · ${p.calls} 调用组`),node('p',p.suggestion));
    if(p.reason_codes.length)box.append(node('p','当前阻止原因：'+p.reason_codes.map(k=>reasonLabels[k]||k).join('；'),'muted'));
    const button=node('button','查看对应算子','secondary');button.onclick=async()=>{try{state.search=p.op;state.offset=0;$('search').value=p.op;showResultTab('types');await groups();}catch(e){message(e.message);}};box.append(button);$('screening-priorities').append(box);
  }
  const s=result.profile_summary,step=a.step.prediction;
  $('step-overview').replaceChildren();
  for(const [label,value] of [['源Step墙钟',s.step_us],['设备活动求和',s.device_sum_us],['设备活动并集',s.busy_us],['Step窗口内设备忙碌',s.step_busy_us],['目标Step基准场景',step?.baseline]]){
    const item=node('p');item.append(node('strong',label+'：'),node('span',value==null?'尚无数据':fmt(value)+' μs'));$('step-overview').append(item);
  }
  if(step)$('step-overview').append(node('p',`源/目标倍率 ${fmt(step.ratio)}×；效率场景范围 ${fmt(step.ratio_low)}–${fmt(step.ratio_high)}×，不是统计置信区间。`,'note'));
  const blockers=node('ul');for(const text of a.step.blockers)blockers.append(node('li',text));$('step-overview').append(blockers);
}
