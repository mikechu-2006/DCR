import numpy as np, pandas as pd
print("step0 CLASSIC 判定权重: perfect=300, great=300, good=200, ok=100, meh=50, miss=0")
print("=> 大P(perfect) 与 300(great) 在 ACC 里权重相同, 两者互换 ACC 变化 = 0")
meta = pd.read_parquet('data/processed/charts_v2_1k_v3.parquet', columns=['beatmap_id', 'countTotal'])
play = pd.read_parquet('data/processed/step1_1k.parquet', columns=['beatmap_id', 'loss'])
d = play.merge(meta, on='beatmap_id', how='left')
print("countTotal 缺失率 = %.2f%%" % (100 * d.countTotal.isna().mean()))
d = d.dropna(subset=['countTotal']); d = d[d.countTotal > 0]
d = d.assign(floor=np.log(250.0 / (300.0 * (d.countTotal.to_numpy() + 1.0))))
g = d.groupby('beatmap_id').agg(n=('loss', 'size'), loss_mean=('loss', 'mean'),
                                floor=('floor', 'first'), N=('countTotal', 'first'))
print("谱面数 = %d, 每谱面 play 数中位 = %d" % (len(g), int(g.n.median())))
print("countTotal: p5=%d p25=%d 中位=%d p75=%d p95=%d" % tuple(np.percentile(g.N, [5, 25, 50, 75, 95]).astype(int)))
print("结构地板 ln(250/(300(N+1))): min=%.3f 中位=%.3f max=%.3f" % (g.floor.min(), g.floor.median(), g.floor.max()))
print("谱面平均 loss:              min=%.3f 中位=%.3f max=%.3f" % (g.loss_mean.min(), g.loss_mean.median(), g.loss_mean.max()))
print("corr(谱面平均loss, 结构地板) = %.4f" % g.loss_mean.corr(g.floor))
gap = g.loss_mean - g.floor
print("gap = 谱面平均loss - 结构地板: 中位=%.3f p10=%.3f p90=%.3f" % (gap.median(), gap.quantile(.1), gap.quantile(.9)))
cap = 1 - np.exp(g.floor); acc = 1 - np.exp(g.loss_mean)
print("ACC 上限 中位=%.5f ; 谱面平均 ACC 中位=%.5f ; 差距中位=%.5f (= %.3f pp)" % (
    cap.median(), acc.median(), (cap - acc).median(), (cap - acc).median() * 100))
print("ALL_DONE")
