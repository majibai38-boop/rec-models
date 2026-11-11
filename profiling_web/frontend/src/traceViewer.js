// Chrome Trace timestamps are microseconds; Perfetto viewport messages use seconds.
export function traceRange(unit, side, eventIndex = 0, whole = false, padding = 1) {
  const kernels = unit?.[side + '_kernels'] || []
  const events = whole ? [...kernels, ...(unit?.[side+'_owners'] || []), ...(unit?.[side+'_scopes'] || [])] : [kernels[eventIndex]]
  const valid = events.filter(e => e && Number.isFinite(e.timestamp_us) && Number.isFinite(e.time_us) && e.time_us >= 0)
  if (!valid.length) return null
  const start = valid.reduce((value,e) => Math.min(value,e.timestamp_us),Infinity)
  const end = valid.reduce((value,e) => Math.max(value,e.timestamp_us+e.time_us),-Infinity)
  const margin = Math.max(10, end-start) * padding
  return {timeStart:(start-margin)/1e6, timeEnd:(end+margin)/1e6, viewPercentage:1}
}

export function viewerUrl(base, range) {
  const params = new URLSearchParams({mode:'embedded'})
  if (range) {
    params.set('visStart', String(Math.round(range.timeStart*1e9)))
    params.set('visEnd', String(Math.round(range.timeEnd*1e9)))
  }
  return base + '#!/?' + params
}

export function waitForViewer(target, origin, signal, host=window, timeoutMs=30000) {
  return new Promise((resolve,reject) => {
    let interval, timeout
    const clean=()=>{clearInterval(interval);clearTimeout(timeout);host.removeEventListener('message',receive);signal?.removeEventListener('abort',abort)}
    const abort=()=>{clean();reject(new Error('时间线加载已取消'))}
    const receive=e=>{if(e.source===target && e.origin===origin && e.data==='PONG'){clean();resolve()}}
    if(signal?.aborted){abort();return}
    host.addEventListener('message',receive)
    signal?.addEventListener('abort',abort,{once:true})
    interval=setInterval(()=>target.postMessage('PING',origin),150)
    timeout=setTimeout(()=>{clean();reject(new Error('查看器启动超时，请重试或检查 Perfetto 安装'))},timeoutMs)
    target.postMessage('PING',origin)
  })
}

export async function sendTrace(target, origin, response, title) {
  if (!response.ok) {
    let detail='Trace 加载失败（'+response.status+'）'
    try {detail=(await response.json()).detail || detail} catch {}
    throw new Error(detail)
  }
  if (response.body) {
    try {
      target.postMessage({perfetto:{stream:response.body,bytesTotal:Number(response.headers.get('content-length'))||0,
        title,keepApiOpen:true}},origin,[response.body])
      return
    } catch (e) {
      if (response.body.locked) throw e
    }
  }
  const size=Number(response.headers.get('content-length'))
  if (!size || size>256*1024*1024) throw new Error('当前浏览器不支持流式传输且 Trace 较大，请使用新版 Chrome / Edge')
  const buffer=await response.arrayBuffer()
  target.postMessage({perfetto:{buffer,title,keepApiOpen:true,shareable:false,downloadable:false}},origin,[buffer])
}
