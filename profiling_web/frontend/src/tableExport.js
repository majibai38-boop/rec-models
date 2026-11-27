import {csvCell} from './viewState.js'

// Use the same immutable query for every page; never export only the visible page.
export async function filteredCsv(api, url, {columns, scope}={}) {
  const rows=[]
  const separator=url.includes('?')?'&':'?'
  for (;;) {
    const data=await api(url+separator+'limit=200&offset='+rows.length)
    if (data.total>100000 || rows.length+data.items.length>100000) throw Error('筛选结果超过 10 万行，请缩小范围或下载完整 Excel')
    rows.push(...data.items)
    if (rows.length>=data.total) break
    if (!data.items.length) throw Error('分页结果不完整，请刷新后重试')
  }
  const fields=columns || [...new Set(rows.flatMap(Object.keys))]
  const lines=[...(scope ? [['Export scope',scope]] : []),fields,...rows.map(r=>fields.map(k=>r[k]))]
  return '\uFEFF'+lines.map(r=>r.map(csvCell).join(',')).join('\r\n')
}
export async function exportCsv(api,url,filename,options) {
  const csv=await filteredCsv(api,url,options)
  const target=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}))
  const a=document.createElement('a');a.href=target;a.download=filename;a.click()
  setTimeout(()=>URL.revokeObjectURL(target),1000)
}
