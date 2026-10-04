/**
 * scikit-learn HistGradientBoostingRegressor prediction from the flat node tables
 * written by scripts/app/export_web_models.py. A line-by-line port of sklearn's
 * _predictor.pyx::_predict_one_from_raw_data:
 *   NaN                       -> missing_go_to_left
 *   categorical, value < 0    -> treated as missing
 *   categorical, in raw bitset -> left; known category not in it -> right; unknown -> missing
 *   numeric                   -> left iff value <= threshold
 */
export interface HGB {
  /** sklearn's internal preprocessor (categorical models only): encoded categoricals first, then numerics */
  pre: { cat_idx: number[]; num_idx: number[]; categories: number[][] } | null
  baseline: number
  roots: number[]
  feature: number[]
  threshold: number[]
  left: number[]
  right: number[]
  missing_left: number[]
  is_cat: number[]
  bitset: number[]
  value: number[]
  bitsets: number[][]
  known_bitsets: number[][] | null
  f_idx_map: number[] | null
}

const inBitset = (bitset: number[], val: number) => ((bitset[val >>> 5] >>> (val & 31)) & 1) === 1

function predictTree(m: HGB, root: number, x: ArrayLike<number>): number {
  let i = root
  for (;;) {
    const f = m.feature[i]
    if (f < 0) return m.value[i]
    const v = x[f]
    let goLeft: boolean
    if (Number.isNaN(v)) {
      goLeft = m.missing_left[i] === 1
    } else if (m.is_cat[i] === 1) {
      const cat = Math.trunc(v) & 0xff // sklearn casts to uint8 (X_BINNED_DTYPE_C)
      if (v < 0) goLeft = m.missing_left[i] === 1
      else if (inBitset(m.bitsets[m.bitset[i]], cat)) goLeft = true
      else if (m.known_bitsets && m.f_idx_map && inBitset(m.known_bitsets[m.f_idx_map[f]], cat)) goLeft = false
      else goLeft = m.missing_left[i] === 1
    } else {
      goLeft = v <= m.threshold[i]
    }
    i = goLeft ? m.left[i] : m.right[i]
  }
}

/** sklearn's ColumnTransformer: OrdinalEncoder on categoricals (unknown -> NaN), then numerics as-is. */
function preprocess(m: HGB, x: ArrayLike<number>): ArrayLike<number> {
  if (!m.pre) return x
  const { cat_idx, num_idx, categories } = m.pre
  const out = new Float64Array(cat_idx.length + num_idx.length)
  cat_idx.forEach((c, k) => {
    const i = Number.isNaN(x[c]) ? -1 : categories[k].indexOf(x[c])
    out[k] = i < 0 ? NaN : i
  })
  num_idx.forEach((c, k) => { out[cat_idx.length + k] = x[c] })
  return out
}

/** Raw prediction (identity link: squared-error and quantile losses). */
export function predictHGB(m: HGB, x: ArrayLike<number>): number {
  const xt = preprocess(m, x)
  let out = m.baseline
  for (const root of m.roots) out += predictTree(m, root, xt)
  return out
}

/** The exporter writes +-inf thresholds as +-Number.MAX_VALUE (JSON has no Infinity); restore them. */
export function restoreInfinities(m: HGB): HGB {
  m.threshold = m.threshold.map((t) => (t === Number.MAX_VALUE ? Infinity : t === -Number.MAX_VALUE ? -Infinity : t))
  return m
}
