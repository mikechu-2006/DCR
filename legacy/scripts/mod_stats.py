import pandas as pd, numpy as np
pd.set_option("display.width", 220)
p = pd.read_parquet("data/processed/plays_4k.parquet")
m = p[p.source == "modern"].copy()
mods = m.mods.fillna("")
def has(s, tok): return s.str.split("+").apply(lambda x: tok in x)
print("modern 4K plays:", len(m))
for tok in ["CL","MR","HD","FL","DT","NC","HT","NF","EZ","HR","SD","PF","SO","AP"]:
    print(f"  {tok:3s} {int(has(mods,tok).sum()):9d}  {has(mods,tok).mean()*100:6.2f}%")
core = mods.str.replace("+CL","",regex=False).str.replace("CL+","",regex=False).str.replace("CL","",regex=False)
print("\nmods after stripping CL (top 15):")
print(core.value_counts().head(15).to_string())
print("\nshare of CL in all scores: %.4f" % has(mods,'CL').mean())
print("rate-change share: DT %.4f NC %.4f HT %.4f" % (has(mods,'DT').mean(), has(mods,'NC').mean(), has(mods,'HT').mean()))
print("plays with rate change and no CL:", int((has(mods,'DT')|has(mods,'NC')|has(mods,'HT') & ~has(mods,'CL')).sum()))
leg = p[p.source=="legacy"]
print("\nlegacy plays:", len(leg), "rate_change share:", round(float(leg.rate_change.mean()),4))
