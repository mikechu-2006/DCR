"""Loopback CPU encoder for the MMA Reform plugin. Run with --help for setup."""
from argparse import ArgumentParser
from hashlib import file_digest, md5
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINS = (
    "http://localhost:24050", "http://127.0.0.1:24050",
    "http://localhost:24061", "http://127.0.0.1:24061",
)
MAX_BODY = 16 * 1024 * 1024
PORT = 24063


def at_rate(raw, rate):
    text = raw.decode("utf-8-sig")
    if not (re.search(r"^Mode:\s*3\s*$", text, re.M)
            and re.search(r"^CircleSize:\s*4(?:\.0+)?\s*$", text, re.M)):
        raise ValueError("Only osu!mania 4K charts can be calculated")
    if rate == 1:
        return raw, text
    lines = []
    section = ""

    def ms(value):
        return str(round(float(value) / rate))

    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            section = line
        elif line.strip() and not line.lstrip().startswith("//"):
            if section == "[TimingPoints]":
                parts = line.split(",")
                parts[0] = format(float(parts[0]) / rate, ".15g")
                if len(parts) < 7 or parts[6].strip() == "1":
                    parts[1] = format(float(parts[1]) / rate, ".15g")
                line = ",".join(parts)
            elif section == "[HitObjects]":
                parts = line.split(",")
                parts[2] = ms(parts[2])
                if int(parts[3]) & 128:
                    end, _, sample = parts[5].partition(":")
                    parts[5] = str(max(int(parts[2]) + 1, int(ms(end)))) + ":" + sample
                line = ",".join(parts)
            elif section == "[Events]" and line.strip().startswith(("2,", "Break,")):
                parts = line.split(",")
                parts[1:3] = [ms(value) for value in parts[1:3]]
                line = ",".join(parts)
            elif section == "[General]" and line.startswith(("AudioLeadIn:", "PreviewTime:")):
                key, value = line.split(":", 1)
                line = f"{key}: {-1 if value.strip() == '-1' else ms(value)}"
            elif section == "[Editor]" and line.startswith("Bookmarks:"):
                line = "Bookmarks: " + ",".join(ms(value) for value in line.split(":", 1)[1].split(",") if value.strip())
        lines.append(line)
    rendered = "\n".join(lines) + "\n"
    return rendered.encode("utf-8"), rendered


class Encoders:
    def __init__(self, args):
        self.args = args
        self.manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        self.lock = threading.Lock()
        self.paths = {
            "DCR": args.d2v_checkpoint,
            "CM3P_ext": args.cm3p_dir / "model.safetensors",
        }
        self.checked = {}
        self.env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                        TORCH_COMPILE_DISABLE="1", TOKENIZERS_PARALLELISM="false",
                        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    def verify_checkpoint(self, method, expected):
        path = self.paths[method]
        if not path.is_file():
            raise ValueError(f"{method} encoder checkpoint is missing. See the CPU service setup.")
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        recorded = self.checked.get(method)
        if not recorded or recorded[0] != identity:
            with path.open("rb") as source:
                self.checked[method] = (identity, file_digest(source, "sha256").hexdigest())
        if self.checked[method][1] != expected:
            raise ValueError(f"{method} encoder checkpoint does not match the plugin release")

    def run(self, command, env=None):
        subprocess.run(command, cwd=ROOT, env=env or self.env, check=True, timeout=480)

    def features(self, method, rate, body, checkpoint, contract):
        cfg = self.manifest["methods"].get(method, {}).get("encoder", {})
        if method not in self.paths or contract != cfg.get("contract") or checkpoint != cfg.get("checkpoint_sha256"):
            raise ValueError("Unknown method or incompatible encoder")
        self.verify_checkpoint(method, checkpoint)
        rendered, _ = at_rate(body, rate)
        import pandas as pd


        with TemporaryDirectory(prefix="dcr-reform-") as directory:
            folder = Path(directory)
            osu = folder / "1.osu"
            out = folder / "embedding.parquet"
            osu.write_bytes(rendered)
            msd_run = subprocess.run([sys.executable, "-m", "deploy.mma.minacalc_feature",
                                      str(osu), str(self.args.minacalc_lib)], cwd=ROOT,
                                     capture_output=True, text=True, check=True, timeout=30, env=self.env)
            overall = float(msd_run.stdout)
            inputs = folder / "sunny.jsonl"
            inputs.write_text(json.dumps({"row_id": "chart", "osu_path": str(osu)}) + "\n")
            sunny_run = subprocess.run([self.args.node, str(ROOT / "scripts/benchmark_sunny.mjs"),
                                        str(inputs), str(ROOT)], cwd=ROOT, capture_output=True,
                                       text=True, check=True, timeout=30, env=self.env)
            output = json.loads(sunny_run.stdout)
            sunny = output.get("sunny_star")
            if sunny is None or not math.isfinite(sunny) or not math.isfinite(overall):
                raise ValueError("Chart did not produce valid difficulty features")
            if method == "DCR":
                ids = folder / "ids.txt"
                ids.write_text("1\n")
                env = dict(self.env, PYTHONPATH=os.pathsep.join(
                    x for x in (str(ROOT / "data2vec_main"), self.env.get("PYTHONPATH", "")) if x))
                self.run([sys.executable, "-m", "d2v_main.extract",
                          "--config", str(ROOT / "data2vec_main/configs/audio_style_a20_product.yaml"),
                          "--ckpt", str(self.args.d2v_checkpoint), "--device", "cpu",
                          "--osu-dir", str(folder), "--ids-file", str(ids), "--no-index-cache",
                          "--batch-size", "16", "--num-workers", "0", "--out", str(out)], env)
            else:
                ids = folder / "ids.csv"
                ids.write_text(f"beatmap_id,checksum\n1,{md5(rendered).hexdigest()}\n")
                self.run([str(self.args.cm3p_python), str(ROOT / "scripts/step0c_chart_content.py"),
                          "--source", "osu-dir", "--ids-file", str(ids), "--osu-dir", str(folder),
                          "--model-repo", str(self.args.cm3p_dir), "--crop", "legacy",
                          "--mania-columns", "off", "--windows", "off", "--audio", "off",
                          "--infer-batch", "8", "--device", "cpu", "--core-audit", "0",
                          "--out", str(out)])
            table = pd.read_parquet(out)
            if len(table) != 1 or int(table.iloc[0].beatmap_id) != 1 or not table.iloc[0].checksum_ok:
                raise RuntimeError("Encoder did not return this chart")
            features = [float(x) for x in table.iloc[0].embedding] + [overall, sunny]
            if len(features) != 514 or not all(math.isfinite(x) for x in features):
                raise RuntimeError("Encoder returned an invalid feature vector")
            return features


def handler(encoders, origins):
    class Handler(BaseHTTPRequestHandler):
        def origin(self):
            host = self.headers.get("Host")
            origin = self.headers.get("Origin")
            if host not in (f"localhost:{PORT}", f"127.0.0.1:{PORT}") or (origin and origin not in origins):
                return False
            return origin or ""

        def send(self, status, value=None):
            body = b"" if value is None else json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            origin = self.origin()
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_OPTIONS(self):
            self.send(204 if self.origin() is not False else 403)

        def do_POST(self):
            url = urlsplit(self.path)
            if self.origin() is False:
                self.send(403, {"error": "Overlay origin is not allowed"})
                return
            if url.path != "/features":
                self.send(404, {"error": "Unknown endpoint"})
                return
            if self.headers.get_content_type() != "application/octet-stream":
                self.send(415, {"error": "Expected raw .osu bytes"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY:
                    raise ValueError("Missing or oversized .osu")
                params = parse_qs(url.query)
                method, expected, rate_text, contract, checkpoint = (
                    params.get(k, [""])[0] for k in ("method", "md5", "rate", "contract", "checkpoint"))
                rate = float(rate_text)
                if not 0.5 <= rate <= 2:
                    raise ValueError("Rate must be between 0.5 and 2")
                self.connection.settimeout(15)
                chart = self.rfile.read(size)
                self.connection.settimeout(None)
                if len(chart) != size or md5(chart).hexdigest() != expected:
                    raise ValueError("Chart checksum does not match the request")
                if not encoders.lock.acquire(blocking=False):
                    self.send(409, {"error": "CPU service is busy. Retry once the current calculation finishes."})
                    return
                try:
                    features = encoders.features(method, rate, chart, checkpoint, contract)
                finally:
                    encoders.lock.release()
                self.send(200, {"method": method, "md5": expected, "rate": rate, "contract": contract,
                                "checkpoint": checkpoint, "features": features})
            except (ValueError, TimeoutError) as error:
                self.send(400, {"error": str(error)})
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                print(f"CPU calculation failed: {error} {getattr(error, 'stderr', '') or ''}", file=sys.stderr, flush=True)
                self.send(503, {"error": "CPU encoder failed. See the local service log."})

    return Handler


def main():
    parser = ArgumentParser(description=__doc__)
    release = ROOT / "data/reform/mma_release"
    manifest = release / "DCR-Reform-tosu/reform-data/manifest.json"
    legacy_manifest = release / "DCR Reform for tosu/reform-data/manifest.json"
    if not manifest.is_file() and legacy_manifest.is_file():
        manifest = legacy_manifest
    parser.add_argument("--manifest", type=Path, default=manifest)
    parser.add_argument("--d2v-checkpoint", type=Path, default=ROOT / "data2vec_main/runs/product/a20_full/checkpoints/last.pt")
    parser.add_argument("--cm3p-dir", type=Path, default=ROOT / ".cache/cm3p_model_26cc694")
    parser.add_argument("--cm3p-python", type=Path, default=ROOT / ".cache/cm3p_venv/bin/python")
    parser.add_argument("--minacalc-lib", type=Path, default=ROOT / "external/minacalc-standalone/build/libminacalc.so")
    parser.add_argument("--node", default="node")
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--origin", action="append", default=[], help="Additional browser origin for a custom tosu port")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    service = ThreadingHTTPServer(("127.0.0.1", PORT), handler(Encoders(args), {*DEFAULT_ORIGINS, *args.origin}))
    print(f"DCR Reform CPU service at http://127.0.0.1:{PORT}", flush=True)
    with service:
        service.serve_forever()


if __name__ == "__main__":
    main()
