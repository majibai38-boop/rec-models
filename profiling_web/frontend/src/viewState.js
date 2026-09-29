// Versioned, bounded display state only. Never contains tokens or file paths.
export const tabKeys = ['overview', 'whole', 'coverage', 'calls', 'distribution']
export function normalizeStep(value) { return String(value || '').trim().replace(/^ProfilerStep#([0-9]+)$/i, '$1') }
const text = value => typeof value === 'string' ? value.slice(0, 200) : ''
const page = value => Number.isInteger(value) && value >= 0 ? Math.min(value, 100000) : 0
const pick = (value, allowed, fallback) => allowed.includes(value) ? value : fallback
const fields = (value, keys) => Object.fromEntries(keys.filter(k => typeof value?.[k] === 'string').map(k => [k, text(value[k])]))
export function cleanView(value = {}) {
  if (!value || typeof value !== 'object') value = {}
  return { tab: pick(value.tab, tabKeys, 'overview'), search: text(value.search),
    sort: pick(value.sort, ['npu_time', 'coverage', 'ratio', 'name'], 'npu_time'), hotspot: value.hotspot === true,
    page: page(value.page), recordPage: page(value.recordPage), detailed: value.detailed === true,
    recordKind: pick(value.recordKind, ['reasons', 'mappings', 'diagnoses'], 'reasons'),
    callSearch: text(value.callSearch), callSort: pick(value.callSort, ['npu_time', 'ratio', 'name'], 'npu_time'),
    callStatus: pick(value.callStatus, ['', 'comparable', 'pending'], ''),
    scope: fields(value.scope, ['op_type', 'device_id', 'core_type']),
    callFilter: fields(value.callFilter, ['related_type', 'group_id', 'device_id', 'core_type']),
    focus: text(value.focus) }
}
export function readLocation(search) {
  const params = new URLSearchParams(search)
  let view = {}
  try { const raw = params.get('view') || '{}'; if (raw.length <= 6000) view = JSON.parse(raw) } catch {}
  return { job: /^[0-9a-f]{32}$/.test(params.get('job') || '') ? params.get('job') : '', view: cleanView(view) }
}
export function viewQuery(job, state) {
  return new URLSearchParams({ job, view: JSON.stringify(cleanView(state)) }).toString()
}
export function readSavedViews(raw) {
  try { const rows = JSON.parse(raw || '[]'); return Array.isArray(rows) ? rows.slice(0, 20).filter(r => r && typeof r.name === 'string').map(r => ({name: r.name.slice(0, 60), view: cleanView(r.view)})) : [] } catch { return [] }
}

export function csvCell(value) {
  let text = typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '')
  const numeric = text.trim() !== '' && Number.isFinite(Number(text))
  if (/^[\t\r\n]/.test(text) || (/^[=+@-]/.test(text.trimStart()) && !numeric)) text = "'" + text
  return '"' + text.replaceAll('"', '""') + '"'
}
