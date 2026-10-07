import re, sys
from pathlib import Path
d = Path("data/raw/extracted/2026_09_01_performance_mania_top_1000")
for f in sorted(d.glob("*.sql")):
    txt = f.read_text(encoding="utf-8", errors="ignore")[:400000]
    i = txt.find("CREATE TABLE")
    j = txt.find("ENGINE=", i)
    block = txt[i:j]
    cols = re.findall(r"\n\s+\`([a-zA-Z_0-9]+)\`\s+([a-zA-Z]+[a-zA-Z0-9()]*)", block)
    print(f"{f.name}  ({f.stat().st_size/1e6:.0f} MB locally)")
    print("   columns:", ", ".join(f"{c}:{t}" for c, t in cols[:14]), ("..." if len(cols) > 14 else ""))
