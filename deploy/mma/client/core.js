const DATA = new URL("../../reform-data/", import.meta.url);
const FEATURE_COUNT = 514;
const RATE_MODS = new Set(["NM", "DT", "NC", "HT", "DC", "HD", "FI", "FL", "NF", "CL"]);
const tiers = ["low", "low/mid", "mid", "mid/high", "high"];
const cache = new Map();
const shards = new Map();
let database;

export const isReformMethod = method => ["Sunny", "DCR", "CM3P_ext"].includes(method);

function loadOnce(map, key, load) {
    if (!map.has(key)) {
        map.set(key, Promise.resolve().then(load).catch(error => {
            map.delete(key);
            throw error;
        }));
    }
    return map.get(key);
}

async function loadFile(path, expected) {
    const response = await fetch(new URL(path, DATA));
    if (!response.ok) throw new Error("Reform assets are unavailable");
    const bytes = await response.arrayBuffer();
    if (expected) {
        const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
        const hex = Array.from(hash, x => x.toString(16).padStart(2, "0")).join("");
        if (hex !== expected) throw new Error("Reform asset checksum does not match");
    }
    return bytes;
}

const json = bytes => JSON.parse(new TextDecoder().decode(bytes));

function loadManifest() {
    return loadOnce(cache, "manifest", async () => {
        const data = json(await loadFile("manifest.json"));
        if (data.schema !== 1) throw new Error("Unsupported Reform asset format");
        return data;
    });
}

export function rawScore(head, features) {
    if (features.length !== FEATURE_COUNT || !Array.from(features).every(Number.isFinite)) {
        throw new Error("Invalid chart features");
    }
    let raw = 0;
    for (let i = 0; i < 512; i++) {
        raw += head.w[i] * ((features[i] - head.mu[i]) / head.sd[i]);
    }
    for (let i = 0; i < 2; i++) {
        raw += head.w[i + 512] * ((features[i + 512] - head.extra_mu[i]) / head.extra_sd[i]);
    }
    return (raw - head.d_mu_raw) / head.d_sd_raw;
}

export function calibrate(raw, { x, y }) {
    if (!Number.isFinite(raw)) throw new Error("No finite difficulty score");
    let lo = 0;
    let hi = x.length - 1;
    if (raw <= x[lo]) return y[lo];
    if (raw >= x[hi]) return y[hi];
    while (hi - lo > 1) {
        const mid = (hi + lo) >> 1;
        if (x[mid] <= raw) lo = mid;
        else hi = mid;
    }
    return y[lo] + (y[hi] - y[lo]) * (raw - x[lo]) / (x[hi] - x[lo]);
}

export function displayDifficulty(value) {
    const [whole, decimals = ""] = value.toString().split(".");
    const hundredths = decimals.padEnd(2, "0").slice(0, 2);
    const tier = tiers[Math.floor(Number(hundredths[0]) / 2)];
    const intro = value < 4;
    const dan = Number(whole) - (intro ? 0 : 4);
    return {
        number: dan + Number(hundredths) / 100,
        intro,
        paletteValue: value / (intro ? 1.5 : 2),
        tierLabel: `${intro ? "Intro" : "Reform"} ${dan} ${tier}`,
    };
}

export function chartRequest(bytes, text, hostIdentity, speedRate, mods) {
    const hostId = hostIdentity.match(/(?:^|\|)id:(\d+)(?=\||$)/)?.[1];
    const fileId = text.match(/^BeatmapID:\s*(\d+)\s*$/m)?.[1];
    const id = Number(hostId || fileId) || null;
    const rate = Number(speedRate);
    return {
        id, md5: md5(bytes), rate,
        supported: /^Mode:\s*3\s*$/m.test(text) && Number.isFinite(rate) && rate >= 0.5 && rate <= 2
            && mods.every(mod => RATE_MODS.has(mod)),
    };
}

// MD5 tracks exact .osu bytes. Web Crypto has no MD5 digest.
const shifts = [[7, 12, 17, 22], [5, 9, 14, 20], [4, 11, 16, 23], [6, 10, 15, 21]];
const constants = Array.from({ length: 64 }, (_, i) => Math.floor(2 ** 32 * Math.abs(Math.sin(i + 1))));

export function md5(bytes) {
    const padded = new Uint8Array(Math.ceil((bytes.byteLength + 9) / 64) * 64);
    padded.set(new Uint8Array(bytes));
    padded[bytes.byteLength] = 0x80;
    const view = new DataView(padded.buffer);
    view.setBigUint64(padded.byteLength - 8, BigInt(bytes.byteLength) * 8n, true);
    const hash = new Int32Array([0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476]);
    for (let block = 0; block < padded.length; block += 64) {
        let [a, b, c, d] = hash;
        for (let i = 0; i < 64; i++) {
            const round = i >> 4;
            const f = round === 0 ? (b & c) | (~b & d)
                : round === 1 ? (d & b) | (~d & c)
                    : round === 2 ? b ^ c ^ d : c ^ (b | ~d);
            const word = round === 0 ? i : round === 1 ? (5 * i + 1) % 16
                : round === 2 ? (3 * i + 5) % 16 : (7 * i) % 16;
            const n = (a + f + constants[i] + view.getInt32(block + 4 * word, true)) | 0;
            const shift = shifts[round][i % 4];
            [a, b, c, d] = [d, (b + ((n << shift) | (n >>> (32 - shift)))) | 0, b, c];
        }
        hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
    }
    const result = new DataView(new ArrayBuffer(16));
    hash.forEach((n, i) => result.setInt32(i * 4, n, true));
    return Array.from(new Uint8Array(result.buffer), x => x.toString(16).padStart(2, "0")).join("");
}

function openDatabase() {
    if (!database) {
        database = new Promise((resolve, reject) => {
            const request = indexedDB.open("dcr-reform", 1);
            request.onupgradeneeded = () => request.result.createObjectStore("features");
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        }).catch(error => {
            database = null;
            throw error;
        });
    }
    return database;
}

async function localFeatures(key, features) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
        const transaction = db.transaction("features", features ? "readwrite" : "readonly");
        const store = transaction.objectStore("features");
        const request = features ? store.put(features, key) : store.get(key);
        transaction.oncomplete = () => resolve(request.result);
        transaction.onabort = () => reject(transaction.error || new Error("Could not save chart features"));
    });
}

async function bundledFeatures(manifest, entry, chart) {
    if (!chart.id || chart.rate !== 1) return null;
    const index = await loadOnce(cache, entry.index.path, async () => {
        const records = json(await loadFile(entry.index.path, entry.index.sha256));
        return new Map(records.map(([id, md5], i) => [`${id}:${md5}`, i]));
    });
    const row = index.get(`${chart.id}:${chart.md5}`);
    if (row === undefined) return null;
    const shard = entry.shards[Math.floor(row / manifest.shard_rows)];
    const data = await loadOnce(shards, shard.path, async () => {
        const bytes = await loadFile(shard.path, shard.sha256);
        if (bytes.byteLength !== shard.bytes) throw new Error("Incorrect Reform shard size");
        return new DataView(bytes);
    });
    if (shards.size > 8) shards.delete(shards.keys().next().value);
    const offset = (row % manifest.shard_rows) * FEATURE_COUNT * 4;
    return Array.from({ length: FEATURE_COUNT }, (_, i) => data.getFloat32(offset + i * 4, true));
}

export async function estimate(method, chart, sunnyStar) {
    if (!chart.supported) return { method, state: "unsupported" };
    const manifest = await loadManifest();
    const entry = manifest.methods[method];
    if (method === "Sunny") {
        return { method, state: "ready", value: calibrate(sunnyStar, entry.knots) };
    }
    const encoder = entry.encoder;
    const key = `${method}:${encoder.contract}:${encoder.checkpoint_sha256}:${chart.md5}:${chart.rate}`;
    const features = await localFeatures(key).catch(() => null)
        ?? await bundledFeatures(manifest, entry, chart);
    return features
        ? { method, state: "ready", value: calibrate(rawScore(entry.head, features), entry.knots) }
        : { method, state: "missing", key, encoder };
}

export async function requestFeatures(result, chart, bytes) {
    const params = new URLSearchParams({
        method: result.method, md5: chart.md5, rate: chart.rate,
        contract: result.encoder.contract, checkpoint: result.encoder.checkpoint_sha256,
    });
    let response;
    try {
        response = await fetch(`http://127.0.0.1:24063/features?${params}`, {
            method: "POST", body: bytes,
            headers: { "Content-Type": "application/octet-stream" },
            signal: AbortSignal.timeout(600_000),
        });
    } catch {
        throw new Error("Local DCR CPU service unavailable or timed out");
    }
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Calculation failed");
    if (payload.md5 !== chart.md5 || payload.rate !== chart.rate || payload.method !== result.method
        || payload.checkpoint !== result.encoder.checkpoint_sha256 || payload.contract !== result.encoder.contract
        || payload.features?.length !== FEATURE_COUNT || !payload.features.every(Number.isFinite)) {
        throw new Error("Calculation returned features for a different chart or encoder");
    }
    await localFeatures(result.key, new Float32Array(payload.features));
}
