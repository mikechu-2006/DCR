import json
import numpy as np
import pandas as pd

from d2v_main.charts import ChartStore
from d2v_main import index_cache

d = pd.read_parquet("../data/processed/chart_content_d2v_holdout.parquet")
print("shape", d.shape)
print("columns", list(d.columns))
print("unique", d.beatmap_id.nunique(), "n_windows", int(d.n_windows.sum()))
print("source", d.source.value_counts().to_dict(), "rev", d.d2v_rev.unique()[:3])
norms = np.array([float(np.linalg.norm(v)) for v in d.embedding])
print("norm min/med/max", norms.min(), np.median(norms), norms.max())
print("dim", len(d.embedding.iloc[0]), "finite", bool(np.isfinite(np.stack(d.embedding)).all()))
store = ChartStore("data/osu_cache")
cur = index_cache._meta(store, 100)
print("current meta", cur)
meta = json.load(open("cache/chart_index_100.json"))
print("cache meta", {k: meta.get(k) for k in cur})
