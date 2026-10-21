// Port-only: the numpy arithmetic nano-vectordb runs, reproduced bit for bit.
//   normalize(a) = a / np.linalg.norm(a, axis=-1, keepdims=True)
//   norm         = sqrt(add.reduce(a * a, axis=-1))
// numpy's add.reduce over a contiguous axis uses pairwise summation
// (numpy/_core/src/umath/loops_utils.h.src, PW_BLOCKSIZE 128, 8 accumulators),
// accumulating in the array's own dtype. float32 steps go through Math.fround:
// one IEEE op in double then rounding to float32 equals the float32 op for
// + - * / and sqrt (double has more than 2*24+2 bits).

const PW_BLOCKSIZE = 128;

/** numpy @TYPE@_pairwise_sum; `round` is Math.fround for float32, identity for float64. */
function pairwiseSum(a: ArrayLike<number>, off: number, n: number, round: (x: number) => number): number {
  if (n < 8) {
    let res = round(-0.0);
    for (let i = 0; i < n; i++) res = round(res + a[off + i]!);
    return res;
  }
  if (n <= PW_BLOCKSIZE) {
    const r = [a[off]!, a[off + 1]!, a[off + 2]!, a[off + 3]!, a[off + 4]!, a[off + 5]!, a[off + 6]!, a[off + 7]!];
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (let j = 0; j < 8; j++) r[j] = round(r[j]! + a[off + i + j]!);
    }
    let res = round(round(round(r[0]! + r[1]!) + round(r[2]! + r[3]!)) + round(round(r[4]! + r[5]!) + round(r[6]! + r[7]!)));
    for (; i < n; i++) res = round(res + a[off + i]!);
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return round(pairwiseSum(a, off, n2, round) + pairwiseSum(a, off + n2, n - n2, round));
}

const id = (x: number) => x;

/** In-place nano-vectordb `normalize` of every float32 row of `m` (rows x dim). */
export function normalizeRowsF32(m: Float32Array, dim: number, rows = m.length / dim): void {
  const sq = new Float32Array(dim);
  for (let r = 0; r < rows; r++) {
    const off = r * dim;
    for (let i = 0; i < dim; i++) sq[i] = Math.fround(m[off + i]! * m[off + i]!);
    // reduce starts from the identity: 0 + pairwise_sum(...)
    const norm = Math.fround(Math.sqrt(Math.fround(0 + pairwiseSum(sq, 0, dim, Math.fround))));
    for (let i = 0; i < dim; i++) m[off + i] = Math.fround(m[off + i]! / norm);
  }
}

/** `normalize` of one vector in float64 (a Python list becomes a float64 array). */
export function normalizeF64(v: ArrayLike<number>): Float64Array {
  const sq = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) sq[i] = v[i]! * v[i]!;
  const norm = Math.sqrt(0 + pairwiseSum(sq, 0, sq.length, id));
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i]! / norm;
  return out;
}

/** `normalize` of one float32 vector, float32 result. */
export function normalizeF32(v: Float32Array): Float32Array {
  const out = Float32Array.from(v);
  normalizeRowsF32(out, out.length, 1);
  return out;
}
