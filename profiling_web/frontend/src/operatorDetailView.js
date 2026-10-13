export const fmt = value => value == null ? '—' : Number(value).toLocaleString('en-US', {maximumFractionDigits: 3})
export const ratioClass = value => value == null ? '' : value >= 2 ? 'ratio-dark' : value >= 1 ? 'ratio-light' : 'ratio-yellow'
export function tensorText(meta, direction = 'input') {
  const shapes = meta?.[direction + '_shapes'] || [], types = meta?.[direction + '_dtypes'] || []
  if (!shapes.length && !types.length) return `${direction === 'input' ? '输入' : '输出'}：未采集`
  return Array.from({length: Math.max(shapes.length, types.length)}, (_, i) =>
    `${direction}[${i}]: ${i < shapes.length ? JSON.stringify(shapes[i]) : '未知 shape'} · ${types[i] || '未知 dtype'}`).join('\n')
}
export function numel(meta, input) {
  const shape = meta?.input_shapes?.[input], dtype = meta?.input_dtypes?.[input]
  if (!Array.isArray(shape) || !dtype || ['none', 'scalar', 'tensorlist'].includes(dtype)) return null
  if (!shape.every(d => Number.isSafeInteger(d) && d >= 0)) return null
  const value = shape.reduce((a, b) => a * b, 1)
  return Number.isSafeInteger(value) ? value : null
}
