'use strict';
const reasonLabels={metadata:'采集元数据缺失或无效',dtype:'精度 / dtype 不支持',parameters:'算子参数语义未建模',implementation:'实现语义待核验',
  unsupported_shape:'Shape / 广播路径未建模',unsupported_operator:'算子模型未覆盖',missing_spec:'缺少硬件规格',
  estimated_spec:'估算规格未获确认',dense_basis:'矩阵稠密口径未确认',precision:'计算精度模式未确认',activity:'内存 / 通信未校准',zero_time:'零时长活动'};
function renderDiagnostics(result){
  const root=$('diagnostics');root.replaceChildren();
  if(!result.diagnostics)root.append(node('p','旧结果没有原因汇总，请复用输入重新分析。','muted'));
  else if(!result.diagnostics.length)root.append(node('p','当前范围没有未建模调用组；预测假设仍需实测验证。','note'));
  for(const g of result.diagnostics||[]){
    const box=node('article',null,'diagnostic');box.append(node('h3',reasonLabels[g.code]||g.code));
    box.append(node('p',`${fmt(g.source_us)} μs · 全部源时间的 ${fmt(g.source_pct)}% · ${g.unit_count} 调用组 / ${g.kernel_count} kernels`));
    const meter=node('meter');meter.max=100;meter.value=g.source_pct||0;meter.setAttribute('aria-label',(reasonLabels[g.code]||g.code)+'未建模时间占比');box.append(meter);
    box.append(node('p',g.suggestion,'note'),node('p',g.operators.join(' · '),'muted'));
    const details=node('details');details.append(node('summary','证据样例（最多3项）'));
    for(const e of g.examples)details.append(node('p',e.op+'：'+e.reason),node('code',e.signature));
    box.append(details);root.append(box);
  }
  const blockers=result.step_blockers??(result.step_prediction?[]:['旧结果未保存逐项原因；查看警告与场景条件。']);
  $('step-blockers').replaceChildren(...(blockers.length?blockers:['已满足当前单流固定间隙场景条件；不是完整调度仿真。']).map(t=>node('li',t)));
}
