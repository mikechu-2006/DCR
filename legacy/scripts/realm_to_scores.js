#!/usr/bin/env node
/**
 * osu!(lazer) client.realm -> score JSON, for legacy/scripts/player_plays.py
 *
 * 用法：
 *   NODE_PATH=/tmp/realmwork/node_modules node legacy/scripts/realm_to_scores.js \
 *       --realm data/player/17598247/client.realm \
 *       --out   data/player/17598247/realm_scores.json
 *
 * 依赖（本机临时装即可，不进仓库）：
 *   mkdir -p /tmp/realmwork && cd /tmp/realmwork
 *   npm_config_cache=/tmp/npm-cache npm install realm
 *
 * 原理：
 *   1. 用「空 schema + migration 回调」打开一份**副本**，从 oldRealm.schema 里
 *      读到这个 realm 文件的真实 schema（类名 / 属性 / 类型）。副本会被迁移，
 *      原始文件不动。
 *   2. 再复制一份原始文件，用刚拿到的 schema 在**原版本号**上打开，读 Score 表。
 *   3. 每条成绩映射成 player_plays.py 认识的 JSON：beatmap.id / checksum /
 *      ruleset_id / mods / accuracy / ended_at …
 *
 * 只读：两个副本用完即删；输入文件不会被写入。
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const Realm = require('realm');

function fail(message) {
  console.error('ERROR: ' + message);
  process.exit(1);
}

function arg(name, fallback = null) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  if (index + 1 >= process.argv.length) fail(name + ' 缺少参数');
  return process.argv[index + 1];
}

function tmpPath(tag) {
  return path.join(os.tmpdir(), 'realm-' + tag + '-' + process.pid + '-' + Date.now() + '.realm');
}

function cleanup(file) {
  for (const suffix of ['', '.lock', '.note', '.management']) {
    try { fs.rmSync(file + suffix, { force: true, recursive: true }); } catch (e) { /* ignore */ }
  }
}

function storedSchemaVersion(realmPath) {
  try {
    const probe = new Realm({ path: realmPath, schema: [], schemaVersion: 0 });
    probe.close();
  } catch (error) {
    const match = /last set version (\d+)/.exec(error.message || '');
    if (match) return Number(match[1]);
    fail('读不到 schema 版本：' + error.message);
  }
  fail('realm 以版本 0 打开成功了，说明文件为空或路径不对');
}

function readSchema(realmPath, version) {
  const copy = tmpPath('schema');
  fs.copyFileSync(realmPath, copy);
  let schema = null;
  try {
    const realm = new Realm({
      path: copy,
      schema: [],
      schemaVersion: version + 1,
      onMigration: (oldRealm) => { schema = oldRealm.schema; },
    });
    realm.close();
  } finally {
    cleanup(copy);
  }
  if (!schema || !schema.length) fail('migration 回调没有拿到 schema');
  return schema;
}

function parseMods(raw) {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed
        .map((mod) => (typeof mod === 'string' ? mod : mod && mod.acronym))
        .filter(Boolean);
    }
  } catch (error) { /* fall through */ }
  return String(raw).split(/[,+]/).map((s) => s.trim()).filter(Boolean);
}

function main() {
  const realmPath = arg('--realm');
  const outPath = arg('--out');
  if (!realmPath || !outPath) fail('需要 --realm <client.realm> 和 --out <json>');
  if (!fs.existsSync(realmPath)) fail('找不到 ' + realmPath);

  const version = storedSchemaVersion(realmPath);
  console.log('realm schema version:', version);
  const schema = readSchema(realmPath, version);
  console.log('classes:', schema.map((c) => c.name).join(', '));

  const copy = tmpPath('read');
  fs.copyFileSync(realmPath, copy);
  const rows = [];
  let skipped = 0;
  try {
    const realm = new Realm({ path: copy, schema, schemaVersion: version });
    const scores = realm.objects('Score');
    console.log('scores:', scores.length);
    for (const score of scores) {
      const beatmap = score.BeatmapInfo;
      if (!beatmap) { skipped += 1; continue; }
      const ruleset = beatmap.Ruleset;
      rows.push({
        source: 'lazer:client.realm',
        id: score.OnlineID || score.LegacyOnlineID || null,
        beatmap: {
          id: beatmap.OnlineID || null,
          checksum: (beatmap.MD5Hash || beatmap.OnlineMD5Hash || '').toLowerCase() || null,
          difficulty: beatmap.DifficultyName || null,
        },
        ruleset_id: ruleset ? ruleset.OnlineID : null,
        mode: ruleset ? ruleset.ShortName : null,
        mods: parseMods(score.Mods),
        total_score: score.TotalScore,
        max_combo: score.MaxCombo,
        accuracy: score.Accuracy,
        pp: score.PP,
        rank: score.Rank,
        passed: score.Rank !== -1,
        ended_at: score.Date ? new Date(score.Date).toISOString() : null,
        is_legacy: score.IsLegacyScore,
      });
    }
    realm.close();
  } finally {
    cleanup(copy);
  }

  fs.mkdirSync(path.dirname(path.resolve(outPath)), { recursive: true });
  fs.writeFileSync(outPath, JSON.stringify(rows, null, 1));
  const byMode = {};
  for (const row of rows) byMode[row.mode || 'unknown'] = (byMode[row.mode || 'unknown'] || 0) + 1;
  console.log('rows:', rows.length, 'skipped(no beatmap):', skipped);
  console.log('by mode:', byMode);
  console.log('->', outPath);
  process.exit(0);
}

main();
