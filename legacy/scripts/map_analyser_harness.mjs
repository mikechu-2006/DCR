#!/usr/bin/env node
/**
 * osumania_map_analyser 特征提取 harness（Node ESM）。
 *
 * 从 external/osumania_map_analyser 的纯计算模块（parser / 估算器 / 键型 / Etterna
 * WASM / Interlude）里逐图提取特征，输出 NDJSON。每个 stage 独立 try/catch，单个
 * 估算器失败只把该 stage 的字段留空，不影响整行。
 *
 * 用法（一般由 scripts/step0i_map_analyser.py 调用）：
 *   node scripts/map_analyser_harness.mjs --list ids.tsv --out out.ndjson \
 *        --plugin-dir 'external/osumania_map_analyser/ManiaMapAnalyser by Leo_Black'
 *
 * --list 每行 'beatmap_id<TAB>/abs/path/osu'，'--shard k/n' 取第 k 片（0 基）。
 * 输出一行一个 JSON；必有 beatmap_id / ok / status，其余字段见文档。
 */

import { closeSync, openSync, readFileSync, writeSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const ETT_VERSIONS = [
    ['068', '0.68.0-Unofficial'],
    ['070', '0.70.0'],
    ['072', '0.72.0'],
    ['0723', '0.72.3'],
    ['074', '0.74.0'],
    ['075', '0.75.0'],
];
const SKILLS = ['overall', 'stream', 'jumpstream', 'handstream', 'stamina', 'jackspeed', 'chordjack', 'technical'];
const ETT_ORDER = ['Overall', 'Stream', 'Jumpstream', 'Handstream', 'Stamina', 'JackSpeed', 'Chordjack', 'Technical'];
const CORE_PATTERNS = ['Stream', 'Chordstream', 'Jacks', 'Coordination', 'Density', 'Wildcard'];

function parseArgs(argv) {
    const out = {};
    for (let i = 0; i < argv.length; i += 1) {
        const a = argv[i];
        if (!a.startsWith('--')) continue;
        const key = a.slice(2);
        const next = argv[i + 1];
        if (next === undefined || next.startsWith('--')) { out[key] = true; continue; }
        out[key] = next; i += 1;
    }
    return out;
}

const args = parseArgs(process.argv.slice(2));
if (!args.list || !args.out) {
    console.error('usage: node map_analyser_harness.mjs --list ids.tsv --out out.ndjson [--plugin-dir DIR] [--shard k/n] [--skip-ett]');
    process.exit(2);
}

const pluginDir = path.resolve(args['plugin-dir'] || 'external/osumania_map_analyser/ManiaMapAnalyser by Leo_Black');
const jsUrl = (rel) => pathToFileURL(path.join(pluginDir, 'js', rel)).href;

const { OsuFileParser } = await import(jsUrl('parser/osuFileParser.js'));
const ett = await import(jsUrl('ett/index.js'));
const patterns = await import(jsUrl('patterns/service.js'));
const interlude = await import(jsUrl('interlude/index.js'));
const { runSunnyEstimatorFromText } = await import(jsUrl('estimator/sunnyEstimator.js'));
const { runDanielEstimatorFromText } = await import(jsUrl('estimator/danielEstimator.js'));
const { runAzusaEstimatorFromText } = await import(jsUrl('estimator/azusaEstimator.js'));
const { runRoxyEstimatorFromText } = await import(jsUrl('estimator/roxyEstimator.js'));
const { runSunnyWindowEstimatorFromText } = await import(jsUrl('estimator/sunnyWindowEstimator.js'));

const lines = readFileSync(args.list, 'utf8').split(/\r?\n/).filter(Boolean);
let rows = lines;
if (args.shard) {
    const parts = String(args.shard).split('/').map(Number);
    rows = lines.filter((_, i) => i % parts[1] === parts[0]);
}

const OPTIONS = {
    speedRate: 1,
    odFlag: null,
    cvtFlag: null,
    withGraph: args['with-graph'] === true,
    enableAnalyzeLN: true,
    extendedEstimationRange: false,
};

const num = (v) => {
    if (v === null || v === undefined || v === '') return null;
    const x = Number(v);
    return Number.isFinite(x) ? x : null;
};

function stage(fn) {
    try {
        return { ok: true, value: fn() };
    } catch (err) {
        return { ok: false, error: String((err && err.message) || err) };
    }
}

async function stageAsync(fn) {
    try {
        return { ok: true, value: await fn() };
    } catch (err) {
        return { ok: false, error: String((err && err.message) || err) };
    }
}

function flattenEtt(prefix, result, row) {
    if (!result || !result.values) return;
    for (let i = 0; i < ETT_ORDER.length; i += 1) {
        row[prefix + '_' + SKILLS[i]] = num(result.values[ETT_ORDER[i]]);
    }
    row[prefix + '_rows'] = num(result.rowCount);
    row[prefix + '_junk'] = result.junkFile ? 1 : 0;
}

function structFeatures(parser, row) {
    const starts = parser.noteStarts;
    const ends = parser.noteEnds;
    const types = parser.noteTypes || [];
    row.keycount = num(parser.columnCount);
    row.od = num(parser.od);
    row.n_notes = starts.length;
    let holds = 0;
    let holdMs = 0;
    for (let i = 0; i < types.length; i += 1) {
        if ((types[i] & 128) !== 0) { holds += 1; holdMs += Math.max(0, (ends[i] || 0) - starts[i]); }
    }
    row.n_holds = holds;
    row.ln_ratio = num(parser.lnRatio);
    if (starts.length) {
        let first = Infinity, last = -Infinity;
        const seen = new Set();
        for (const t of starts) {
            if (t < first) first = t;
            if (t > last) last = t;
            seen.add(t);
        }
        row.duration_s = (last - first) / 1000;
        row.n_rows = seen.size;
        row.chord_ratio = seen.size ? starts.length / seen.size : null;
        row.notes_per_sec = row.duration_s > 0 ? starts.length / row.duration_s : null;
        row.hold_ms_per_note = starts.length ? holdMs / starts.length : null;
    }
    const tps = parser.timingPoints || [];
    if (tps.length) {
        const beats = tps.map((tp) => 60000 / tp[1]).filter((b) => Number.isFinite(b) && b > 0 && b < 1000);
        beats.sort((a, b) => a - b);
        row.bpm_median = beats.length ? beats[Math.floor(beats.length / 2)] : null;
        row.bpm_max = beats.length ? beats[beats.length - 1] : null;
        row.n_timing_points = tps.length;
    }
    row.n_breaks = (parser.breaks || []).length;
    row.break_s = (parser.breaks || []).reduce((s, br) => s + Math.max(0, br[1] - br[0]), 0) / 1000;
}

function patternFeatures(report, row) {
    if (!report) return;
    row.patt_ln_percent = num(report.LNPercent);
    row.patt_hb_ratio = num(report.HBRowRatio);
    row.patt_sv_amount = num(report.SVAmount);
    row.patt_duration_s = num(report.Duration) / 1000;
    row.patt_category = report.Category == null ? null : String(report.Category);
    row.patt_mode_tag = report.ModeTag == null ? null : String(report.ModeTag);
    const clusters = Array.isArray(report.Clusters) ? report.Clusters : [];
    row.patt_n_clusters = clusters.length;
    for (const p of CORE_PATTERNS) {
        const cs = clusters.filter((c) => c.Pattern === p);
        const key = 'patt_' + p.toLowerCase();
        row[key + '_n'] = cs.length;
        row[key + '_amount_s'] = cs.reduce((s, c) => s + (num(c.Amount) || 0), 0) / 1000;
        row[key + '_bpm_max'] = cs.length ? Math.max(...cs.map((c) => num(c.BPM) || 0)) : 0;
        row[key + '_importance'] = cs.length
            ? Math.max(...cs.map((c) => (num(c.Amount) || 0) * (num(c.RatingMultiplier) || 0) * (num(c.BPM) || 0)))
            : 0;
        const wsum = cs.reduce((s, c) => s + (num(c.Amount) || 0), 0);
        row[key + '_mult'] = wsum > 0
            ? cs.reduce((s, c) => s + (num(c.RatingMultiplier) || 0) * (num(c.Amount) || 0), 0) / wsum
            : null;
    }
    row.patt_n_specific = clusters.reduce((s, c) => s + (Array.isArray(c.SpecificTypes) ? c.SpecificTypes.length : 0), 0);
}

async function graphFeatures(value, row) {
    if (!value || !value.graph) return;
    const gv = Array.from(value.graph.values || []).map(Number).filter(Number.isFinite);
    if (!gv.length) return;
    const sorted = [...gv].sort((a, b) => a - b);
    const q = (p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))))];
    const mean = gv.reduce((s, v) => s + v, 0) / gv.length;
    const max = sorted[sorted.length - 1];
    row.graph_mean = mean;
    row.graph_p50 = q(0.5);
    row.graph_p90 = q(0.9);
    row.graph_max = max;
    row.graph_sd = Math.sqrt(gv.reduce((s, v) => s + (v - mean) * (v - mean), 0) / gv.length);
    row.graph_spike = mean > 0 ? max / mean : null;
    const hot = gv.filter((v) => max > 0 && v >= 0.8 * max).length;
    row.graph_hot_frac = hot / gv.length;
}


async function analyzeOne(beatmapId, filePath) {
    const row = { beatmap_id: beatmapId, ok: 0, status: null };
    let text;
    try {
        text = readFileSync(filePath, 'utf8');
    } catch (err) {
        row.error = 'read: ' + String((err && err.message) || err);
        return row;
    }
    const parser = new OsuFileParser(text);
    parser.process();
    row.status = parser.status;
    if (parser.status !== 'OK' || !parser.columns.length) {
        row.error = 'parse: status=' + parser.status + ' notes=' + parser.columns.length;
        return row;
    }
    if (args['only-graph']) {
        const g = stage(() => runSunnyEstimatorFromText(text, Object.assign({}, OPTIONS, { withGraph: true }), parser));
        if (g.ok && g.value) graphFeatures(g.value, row); else if (!g.ok) row.err_sunny = g.error;
        row.ok = 1;
        return row;
    }
    const st = stage(() => structFeatures(parser, row));
    if (!st.ok) row.err_struct = st.error;

    const sun = stage(() => runSunnyEstimatorFromText(text, Object.assign({}, OPTIONS, { withPpMetrics: true }), parser));
    if (sun.ok && sun.value) graphFeatures(sun.value, row);
    if (sun.ok && sun.value) {
        row.sunny_star = num(sun.value.star);
        const m = sun.value.ppMetrics;
        if (m) {
            row.sunny_variety = num(m.variety);
            row.sunny_acc_scalar = num(m.accScalar);
            row.sunny_spikiness = num(m.spikiness);
            row.sunny_switches = num(m.switches);
        }
    } else if (!sun.ok) row.err_sunny = sun.error;

    const dan = stage(() => runDanielEstimatorFromText(text, OPTIONS, parser));
    if (dan.ok && dan.value) row.daniel_star = num(dan.value.star); else if (!dan.ok) row.err_daniel = dan.error;

    const az = stage(() => runAzusaEstimatorFromText(text, Object.assign({}, OPTIONS, { forceSunnyReferenceHo: true }), parser));
    if (az.ok && az.value) {
        row.azusa_star = num(az.value.star);
        row.azusa_numeric = num(az.value.numericDifficulty);
        row.azusa_raw_numeric = num(az.value.rawNumericDifficulty);
    } else if (!az.ok) row.err_azusa = az.error;

    const rox = stage(() => runRoxyEstimatorFromText(text, OPTIONS, parser));
    if (rox.ok && rox.value) {
        row.roxy_star = num(rox.value.star);
        row.roxy_numeric = num(rox.value.numericDifficulty);
        row.roxy_raw_numeric = num(rox.value.rawNumericDifficulty);
    } else if (!rox.ok) row.err_roxy = rox.error;

    const win = stage(() => runSunnyWindowEstimatorFromText(text, OPTIONS, parser));
    if (win.ok && win.value) {
        row.sunnywin_ln_star = num(win.value.lnStar);
        const tp = win.value.typePercentageData;
        if (tp && typeof tp === 'object') row.sunnywin_n_types = Object.keys(tp).length;
    } else if (!win.ok) row.err_sunnywin = win.error;

    const itl = await stageAsync(() => interlude.calculateInterludeStar(parser, 1, null));
    if (itl.ok) row.interlude_star = num(itl.value); else row.err_interlude = itl.error;

    const pat = stage(() => patterns.analyzePatternFromText(text));
    if (pat.ok) patternFeatures(pat.value && pat.value.report, row); else row.err_pattern = pat.error;

    if (!args['skip-ett']) {
        for (const pair of ETT_VERSIONS) {
            const prefix = 'ett_' + pair[0];
            const r = await stageAsync(() => ett.analyzeEtternaFromText(text, { musicRate: 1, scoreGoal: 0.93, etternaVersion: pair[1] }));
            if (r.ok) flattenEtt(prefix, r.value, row); else row['err_' + prefix] = r.error;
        }
    }

    row.ok = 1;
    return row;
}

const fd = openSync(args.out, 'w');
const t0 = Date.now();
for (let i = 0; i < rows.length; i += 1) {
    const parts = rows[i].split('\t');
    const id = Number(parts[0]);
    let row;
    try {
        row = await analyzeOne(id, parts[1]);
    } catch (err) {
        row = { beatmap_id: id, ok: 0, error: String((err && err.message) || err) };
    }
    writeSync(fd, JSON.stringify(row) + '\n');
    if ((i + 1) % 50 === 0 || i + 1 === rows.length) {
        const dt = (Date.now() - t0) / 1000;
        const rate = (i + 1) / dt;
        const eta = ((rows.length - i - 1) / Math.max(rate, 1e-9) / 60).toFixed(1);
        console.error('[harness] ' + (i + 1) + '/' + rows.length + '  ' + rate.toFixed(2) + ' charts/s  eta ' + eta + ' min');
    }
}
closeSync(fd);
console.error('[harness] wrote ' + args.out + ' (' + rows.length + ' rows, ' + ((Date.now() - t0) / 1000).toFixed(1) + 's)');
