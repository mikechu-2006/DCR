import numpy as np, pandas as pd
# 用已测得的数字做一次精确的方差分解（全部来自本次会话的实测）
# 来源：diag_step2_structure / diag_step2_vs_additive / _tmp_target
tau2      = 1.177    # 测试 cell 均值中去噪后的交互方差（同seed不同数据，cell RMSE 1.805, sd 2.106）
var_cell  = 2.1063**2
noise_in_cellmean = 0.0958   # E[sigma^2/n]
print("=== 交互方差 tau^2 的构成（测试集 cell 级）===")
print(f"  cell 均值的总方差            = {var_cell:.4f}")
print(f"  其中测量噪声 E[sigma^2/n]    = {noise_in_cellmean:.4f}")
print(f"  => 真实交互方差 tau^2        = {tau2:.4f}  (占 cell 均值方差的 {tau2/var_cell*100:.1f}%)")
print()
print("=== 各模型在这块 tau^2 上吃到了多少 ===")
print(f"{'模型':<26}{'cell RMSE':>10}{'MSE':>9}{'相对加性减少':>13}{'占 tau^2':>10}")
rows = [("饱和加性(玩家+谱面)", 1.5970), ("mirt dim2", 1.5741), ("mirt dim3", 1.5665),
        ("mirt dim4", 1.5630), ("mirt dim8/16", 1.5561), ("mirt F1(+ln pc)", 1.5263)]
base = 1.5970**2
for nm, r in rows:
    mse = r*r
    d = base - mse
    print(f"{nm:<26}{r:>10.4f}{mse:>9.4f}{d:>13.4f}{d/tau2*100:>9.1f}%")
print()
print("读法：把 cell 均值当作预测目标时，'加性模型'已经吃掉 1.5970^2 = %.3f 的 MSE；" % base)
print("      真正剩下的可吃部分只有 tau^2 = %.3f，而 dim16 只吃掉了 %.3f（%.0f%%）。" % (tau2, base-1.5561**2, (base-1.5561**2)/tau2*100))
print()
print("=== 也就是说 ===")
print(f"  剩余可吃 = tau^2 - (吃到) = {tau2 - (base-1.5561**2):.4f}")
print(f"  如果全部吃到，cell RMSE 会从 1.5561 降到 {np.sqrt(1.5561**2 - (tau2-(base-1.5561**2))):.4f}")
print(f"  play 级 RMSE 会从 1.4577 降到约 {np.sqrt(0.8524 + tau2 - (base-1.5561**2)):.4f}")
