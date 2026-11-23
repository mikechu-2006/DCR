#!/usr/bin/env node
// Run Koishi v5.1 on a list of DCR charts. Only 11 <= d < 17 is valid.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [listPath, outPath, shardArg = "0/1", koishiPath = ".cache/methods/Koishi/koishi.mjs"] = process.argv.slice(2);
if (!listPath || !outPath) {
  console.error("usage: node run_koishi_signals.mjs <list.tsv> <out.ndjson> [k/n] [koishi.mjs]");
  process.exit(2);
}
const [k, n] = shardArg.split("/").map(Number);
const { predictDetails } = await import(pathToFileURL(path.resolve(koishiPath)).href);

const lines = fs.readFileSync(listPath, "utf8").split(/\r?\n/).filter(Boolean);
const rows = lines.filter((_, i) => i % n === k);
const fd = fs.openSync(outPath, "w");
const t0 = Date.now();
let ok = 0;
for (let i = 0; i < rows.length; i += 1) {
  const parts = rows[i].split("\t");
  const bid = Number(parts[0]);
  const file = parts[1];
  const rec = { beatmap_id: bid, method: "koishi_v5", version: "6828d61",
                raw: null, valid: false, reason: null, error: null, runtime_ms: null };
  const t = Date.now();
  try {
    const text = fs.readFileSync(file, "utf8");
    const d = predictDetails(text, 1);
    rec.raw = Number(d.constant);
    rec.valid = true;
    rec.reason = "OK";
    ok += 1;
  } catch (err) {
    const msg = String((err && err.message) || err);
    rec.error = msg;
    rec.reason = msg.startsWith("BelowScope") ? "BelowScope"
      : msg.startsWith("AboveScope") ? "AboveScope"
      : msg.startsWith("TooShort") ? "TooShort"
      : msg.startsWith("UnsupportedLN") ? "LN"
      : "RunnerError";
  }
  rec.runtime_ms = Date.now() - t;
  fs.writeSync(fd, JSON.stringify(rec) + "\n");
  if ((i + 1) % 200 === 0 || i + 1 === rows.length) {
    const dt = (Date.now() - t0) / 1000;
    console.error("[koishi] shard " + k + "/" + n + " " + (i + 1) + "/" + rows.length +
                  " ok=" + ok + " " + ((i + 1) / dt).toFixed(2) + " charts/s");
  }
}
fs.closeSync(fd);
console.error("[koishi] wrote " + outPath + " (" + rows.length + " rows, ok=" + ok + ")");
