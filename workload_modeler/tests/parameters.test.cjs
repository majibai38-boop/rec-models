const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

function setup(){
  const elements={};
  function node(tag,text){return {tag,text,style:{},children:[],append(...xs){this.children.push(...xs);},prepend(x){this.children.unshift(x);elements[x.id]=x;},replaceChildren(...xs){this.children=xs;}};}
  elements['tab-overview']=node('div');
  const ctx=vm.createContext({node,$:id=>elements[id]});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/assessment.js'),'utf8'),ctx);
  const text=n=>[n.text||'',...n.children.map(text)].join(' ');
  return {ctx,elements,text:()=>text(elements['tab-overview'])};
}
test('old results ask for reanalysis and do not display zero parameters',()=>{
  const x=setup();x.ctx.renderParameters(undefined);
  assert.match(x.text(),/旧结果/);assert.doesNotMatch(x.text(),/0\.000/);
});
test('parameter panel separates unknown counts and ambiguous matrices',()=>{
  const x=setup();x.ctx.renderParameters({candidate_parameters:36,dense_candidates:36,embedding_candidates:null,ambiguous_matrix_elements:100,note:'估算',groups:[]});
  assert.match(x.text(),/未识别/);assert.match(x.text(),/不计入合计/);assert.match(x.text(),/36/);
  x.ctx.renderParameters(undefined);
  assert.equal(x.elements['tab-overview'].children.length,1);
  assert.doesNotMatch(x.text(),/候选参数合计/);
});
