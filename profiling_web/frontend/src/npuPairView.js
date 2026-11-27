// Shared presentation helpers for the NPU-pair views.
export const PAIR_COLORS = {
  'dark-green': '#197746', green: '#8fcf9b', yellow: '#f3df8c', 'dark-yellow': '#f0b429', '': '#9aa5a0',
}

// Four classes per CV type, mirroring profiling_tools.npu_pair.ratio_color.
export function colorOf(ratio, cv, thresholds) {
  const limit = cv === 'Cube' ? thresholds?.cube : cv === 'Vector' ? thresholds?.vector : null
  if (ratio == null || !(Number(limit) > 0)) return ''
  // Round the derived bounds so 1.1 * 1.5 compares as 1.65, not 1.6500000000000001.
  const upper = Number((limit * 1.5).toFixed(9)), lower = Number((limit * 0.5).toFixed(9))
  if (ratio >= upper) return 'dark-green'
  if (ratio >= limit) return 'green'
  if (ratio >= lower) return 'yellow'
  return 'dark-yellow'
}

// Element count of one input tensor from "32,64;64,128" / "FLOAT16;FLOAT16".
// Tolerates the wrappers profilers print ("torch.Size([32, 64])") and a trailing comma; a
// dynamic or unknown dimension (-1, ?, 0) cannot be turned into an element count.
export function numelOf(shapes, dtypes, index = 0) {
  const parts = String(shapes || '').split(';')
  const raw = String(parts[index] ?? parts[0] ?? '').trim()
  const dtype = String(String(dtypes || '').split(';')[index] ?? '').trim().toLowerCase()
  if (!raw || ['none', 'scalar', 'tensorlist'].includes(dtype)) return null
  const cleaned = raw.replace(/torch\.size/gi, '').replace(/[\[\](){}]/g, '')
  const pieces = cleaned.split(',').map(piece => piece.trim()).filter(Boolean)
  if (!pieces.length) return null
  const dims = pieces.map(Number)
  if (dims.some(value => !Number.isSafeInteger(value) || value < 1)) return null
  const total = dims.reduce((a, b) => a * b, 1)
  return Number.isSafeInteger(total) ? total : null
}

// How many input tensors the group carries, so the chart can offer an input selector.
export function inputCountOf(rows) {
  return Math.max(1, ...rows.map(row => String(row['输入 Shape'] || '').split(';').filter(Boolean).length))
}
