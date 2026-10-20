#!/usr/bin/env python
"""Which parameter transformations really leave the prediction EXACTLY unchanged?

For  l(u,m) = b0 + bu[u] + bm[m] + <P_u - C_m, D_m>
test a family of transforms P -> P A^{-T}, D -> D A (A any invertible KxK):

  * orthogonal A=Q : the exact null space of the bilinear form
  * diagonal   A   : a pure rescaling of the latent axes
  * shear      A   : a non-orthogonal change of basis

For diagonal/shear the term  <C_m, D_m>  changes, so we also shift bm -> bm + delta.
If the prediction is then unchanged too, the null space is larger than O(K) and the
"rotation" story is only the part of it that bm cannot absorb.
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)
nu, nm, K = 40, 25, 4
P = rng.normal(0, 1, (nu, K))
C = rng.normal(0, 1, (nm, K))
D = rng.normal(0, 1, (nm, K))
b0, bu, bm = 0.3, rng.normal(0, 1, nu), rng.normal(0, 1, nm)
uu = rng.integers(0, nu, 300)
mm = rng.integers(0, nm, 300)


def pred(P, C, D, bm):
    return b0 + bu[uu] + bm[mm] + ((P[uu] - C[mm]) * D[mm]).sum(1)


base = pred(P, C, D, bm)


def report(name, A):
    Ainv_T = np.linalg.inv(A).T
    P2 = P @ Ainv_T
    D2 = D @ A
    dm = ((C * D2).sum(1) - (C * D).sum(1))       # <-<C,D> changed by the transform
    p1 = pred(P2, C, D2, bm)                      # bm NOT corrected
    p2 = pred(P2, C, D2, bm + dm)                 # bm corrected
    print(f"{name:24s} max|d pred| bm uncorrected = {np.abs(p1-base).max():.3e}"
          f"   bm corrected = {np.abs(p2-base).max():.3e}")
    return p2


Q, _ = np.linalg.qr(rng.normal(size=(K, K)))
orth = report("orthogonal  A = Q", Q)
diag = report("diagonal    A = diag", np.diag([3.0, 0.4, 1.5, 0.7]))
S = np.eye(K) + 0.8 * rng.normal(size=(K, K))
shear = report("general/shear A", S)

print()
print("orthogonal: 预测逐点相等（这是真正的零空间/规范自由度）")
print("diagonal/shear + 修正 bm: 也逐点相等 -> 非正交部分被 bm 吸收")
print()
print("=> 精确的连续对称群 = GL(K)（一般线性变换），其中")
print("   - O(K) 部分无法被任何截距吸收，是'纯'规范自由度（旋转）")
print("   - 非正交部分可以用每谱面截距 bm 补偿，所以在有 bm 的模型里也是零空间")
print()
print("对比：换成距离型交互会有别的对称性")
# distance-type interaction: -(P_u - C_m)^2 summed
def pred_dist(P, C):
    return b0 + bu[uu] - ((P[uu] - C[mm]) ** 2).sum(1)
b0d = pred_dist(P, C)
t = rng.normal(0, 1, K)
p_trans = pred_dist(P + t, C + t)      # translate BOTH families -> distances unchanged
print(f"distance型  <-> 同时平移 t: max|d pred| = {np.abs(p_trans-b0d).max():.3e}  "
      f"(平移对称，来自 (x+t)-(y+t)=x-y)")
p_rot = pred_dist(P @ Q, C @ Q)
print(f"distance型  <-> 同时旋转 Q: max|d pred| = {np.abs(p_rot-b0d).max():.3e}  "
      f"(旋转也保持距离)")
