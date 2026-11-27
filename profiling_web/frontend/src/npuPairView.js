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

// Element count of the first input tensor from "32,64;64,128" / "FLOAT16;FLOAT16".
export function numelOf(shapes, dtypes) {
  const shape = String(shapes || '').split(';')[0].trim()
  const dtype = String(dtypes || '').split(';')[0].trim().toLowerCase()
  if (!shape || ['none', 'scalar', 'tensorlist'].includes(dtype)) return null
  const dims = shape.split(',').map(part => Number(part.trim()))
  if (!dims.length || dims.some(value => !Number.isSafeInteger(value) || value < 0)) return null
  const total = dims.reduce((a, b) => a * b, 1)
  return Number.isSafeInteger(total) ? total : null
}
