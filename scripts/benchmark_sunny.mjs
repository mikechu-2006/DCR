#!/usr/bin/env node
// Headless Sunny star for a JSONL manifest of {row_id, osu_path}.
// Uses the same plugin commit/entry point that produced DCR's sunny_star.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [manifestPath, repoRoot] = process.argv.slice(2);
if (!manifestPath || !repoRoot) {
  console.error("usage: benchmark_sunny.mjs <manifest.jsonl> <repo-root>");
  process.exit(2);
}
const estimatorPath = path.join(
  repoRoot,
  "external/osumania_map_analyser/ManiaMapAnalyser by Leo_Black/js/estimator/sunnyEstimator.js",
);
const { runSunnyEstimatorFromText } = await import(pathToFileURL(estimatorPath).href);

const lines = fs.readFileSync(manifestPath, "utf8").split("\n");
for (const line of lines) {
  if (!line.trim()) continue;
  let rec;
  try {
    rec = JSON.parse(line);
  } catch (err) {
    console.log(JSON.stringify({ error: "bad json: " + err.message }));
    continue;
  }
  try {
    const osuPath = path.isAbsolute(rec.osu_path) ? rec.osu_path : path.join(repoRoot, rec.osu_path);
    const osuText = fs.readFileSync(osuPath, "utf8");
    const result = runSunnyEstimatorFromText(osuText, { speedRate: 1.0 });
    console.log(JSON.stringify({
      row_id: rec.row_id,
      sunny_star: Number(result.star),
      sunny_ln_ratio: Number(result.lnRatio),
      sunny_columns: Number(result.columnCount),
      error: null,
    }));
  } catch (err) {
    console.log(JSON.stringify({ row_id: rec.row_id, error: err.name + ": " + err.message }));
  }
}
