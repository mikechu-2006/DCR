"""Build a tosu plugin ZIP with the frozen Reform release assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MMA = ROOT / "external/osumania_map_analyser"
PLUGIN = "ManiaMapAnalyser by Leo_Black"
RELEASE_DIR = "DCR-Reform-tosu"
BASE = "448ed7c789af3d715acfb8370e41244cc3c00865"
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/processed"
REFORM = ROOT / "data/reform"
EXTRAS = ["msd_overall", "sunny_star"]
SHARD_ROWS = 1024


def digest(path, algorithm="sha256"):
    with path.open("rb") as file:
        return hashlib.file_digest(file, algorithm).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def make_release(out, checkpoint_sha):
    out.mkdir()
    whitelist = pd.read_csv(ROOT / "manifests/whitelist_1k.csv", dtype={"checksum": str})
    ids = whitelist.set_index("beatmap_id", verify_integrity=True)["checksum"].str.lower()
    if not all(re.fullmatch("[0-9a-f]{32}", value) for value in ids):
        raise ValueError("invalid whitelist MD5")
    signals = pd.read_parquet(DATA / "rank_signals_1k.parquet",
                              columns=["beatmap_id", *EXTRAS]).set_index("beatmap_id", verify_integrity=True)
    anchor_map = pd.read_json(REFORM / "feature_id_map.json").set_index("feature_beatmap_id", verify_integrity=True)
    anchor_signals = pd.read_parquet(REFORM / "signals.parquet").set_index("bid", verify_integrity=True)
    calibration = json.loads((REFORM / "calibration.json").read_text())
    for key, fname in (("anchor_manifest_sha256", "anchors.jsonl"), ("raw_scores_sha256", "raw_scores.csv")):
        if calibration[key] != digest(REFORM / fname):
            raise ValueError(f"Reform calibration does not match {fname}")
    provenance = calibration["raw_score_inputs_sha256"]
    for key, fname in (("anchor_feature_map", "feature_id_map.json"), ("signals", "signals.parquet")):
        if provenance[key] != digest(REFORM / fname):
            raise ValueError(f"Reform raw scores do not match {fname}")
    methods = {"Sunny": {"id": "SunnyRework", "knots": calibration["methods"]["Sunny"]["calibration"]}}
    specs = (
        ("CM3P_ext", "cm3p", "cm3p_linear_extra", "chart_content_cm3p_local",
         "chart_content_cm3p_legacy_cpu", "RM_L",
         digest(ROOT / ".cache/cm3p_model_26cc694/model.safetensors")),
        ("DCR", "a20d", "d2v_a20d_linear_extra", "chart_content_d2v_a20d_prod",
         "chart_content_d2v_a20d_product_anchors", None, checkpoint_sha),
    )
    for method, prefix, model_name, content, anchor_content, alias, encoder_sha in specs:
        model_path = ROOT / "deploy/models" / f"{model_name}.json"
        model = json.loads(model_path.read_text())
        content_path = DATA / f"{content}.parquet"
        anchor_path = REFORM / f"{anchor_content}.parquet"
        if model["content_md5"] != digest(content_path, "md5"):
            raise ValueError(f"{method} model does not match the vector table")
        if (model["model_type"], model["content_dim"], model["extra_names"]) != ("linear_extra", 512, EXTRAS):
            raise ValueError(f"unsupported {method} head")
        cal_entry = calibration["methods"][method]
        if alias:
            cal_entry = cal_entry["models"][alias]
        if cal_entry["model_id"] != (alias or model["model_id"]):
            raise ValueError(f"{method} calibration belongs to a different head")
        cal_model = model_path if not alias else ROOT / "results/benchmark/models/RM_L_full.pt"
        if provenance["cm3p_content" if alias else "d2v_content"] != digest(anchor_path):
            raise ValueError(f"{method} anchor embeddings do not match calibration")
        if provenance["rm_model" if alias else "dcr_model"] != digest(cal_model):
            raise ValueError(f"{method} calibration does not match its input head")
        if alias:
            # The published JSON is the deployment export of this exact RM-L checkpoint.
            import torch
            checkpoint = torch.load(cal_model, map_location="cpu", weights_only=False)
            for key in ("w", "mu", "sd", "extra_mu", "extra_sd", "d_mu_raw", "d_sd_raw"):
                if not np.allclose(np.asarray(checkpoint[key]), np.asarray(model[key]), rtol=0, atol=1e-9):
                    raise ValueError(f"RM-L export mismatch on {key}")
        rows = {}
        counts = {"base": 0, "reform": 0, "omitted": 0}
        for label, path, checksums, extras in (
            ("base", content_path, ids, signals),
            ("reform", anchor_path, anchor_map["checksum"], anchor_signals),
        ):
            for row in pd.read_parquet(path).itertuples(index=False):
                bid = int(row.beatmap_id)
                if bid not in checksums or bid not in extras.index or not row.checksum_ok:
                    counts["omitted"] += 1
                    continue
                if label == "reform" and anchor_map.loc[bid, "is_manual_split"]:
                    continue  # IDs assigned for offline encoding have no game counterpart.
                md5 = checksums.loc[bid]
                features = np.asarray([*row.embedding, *extras.loc[bid, EXTRAS]], dtype="<f4")
                if features.shape != (514,) or not np.isfinite(features).all():
                    raise ValueError(f"invalid features for {method} {bid}")
                key = (bid, md5)
                if key not in rows:
                    rows[key] = features
                    counts[label] += 1
        keys = sorted(rows)
        shards = []
        for i in range(0, len(keys), SHARD_ROWS):
            path = out / f"{prefix}-{len(shards)}.bin"
            np.stack([rows[key] for key in keys[i:i+SHARD_ROWS]]).tofile(path)
            shards.append({"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
        index = out / f"{prefix}-index.json"
        write_json(index, keys)
        encoder = {"checkpoint_sha256": encoder_sha,
                   "contract": "cm3p-legacy-loader-cpu-v1" if alias else "d2v-a20d-product-cpu-v1"}
        methods[method] = {
            "id": model["model_id"], "knots": cal_entry["calibration"],
            "encoder": encoder, "count": len(keys), "coverage": counts, "shards": shards,
            "index": {"path": index.name, "sha256": digest(index)},
            "head": {key: model[key] for key in ("w", "mu", "sd", "extra_mu", "extra_sd", "d_mu_raw", "d_sd_raw")},
        }
        print(method, counts, "total", len(keys))
    manifest = {"schema": 1, "asset_base": BASE, "shard_rows": SHARD_ROWS,
                "calibration_sha256": digest(REFORM / "calibration.json"),
                "methods": methods}
    manifest["release"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    write_json(out / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/reform/mma_release")
    parser.add_argument("--d2v-checkpoint-sha256", default="9039f55a242a2af53dec05e09af62472dbf0ae594e88f82fc67d768e672b4173")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{64}", args.d2v_checkpoint_sha256):
        parser.error("invalid checkpoint SHA-256")
    out = args.out.resolve()
    if out.exists():
        parser.error(f"output exists; use a new destination: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mma-", dir=out.parent) as tmp:
        stage = Path(tmp)
        # Use the fixed source commit to keep the release reproducible.
        with subprocess.Popen(["git", "-C", str(MMA), "archive", BASE, PLUGIN, "LICENSE"],
                              stdout=subprocess.PIPE) as archive:
            unpack = subprocess.run(["tar", "-xf", "-", "-C", str(stage)], stdin=archive.stdout)
            if archive.wait() or unpack.returncode:
                raise RuntimeError("could not stage the pinned MMA tree")
        subprocess.run(["patch", "-p1", "--batch", "-i", str(HERE / "mma.patch")], cwd=stage, check=True)
        plugin = stage / PLUGIN
        shutil.copytree(HERE / "client", plugin / "js/reform")
        shutil.copy2(stage / "LICENSE", plugin / "LICENSE")
        shutil.copy2(HERE / "README.md", plugin / "README.md")
        make_release(plugin / "reform-data", args.d2v_checkpoint_sha256)
        plugin.rename(stage / RELEASE_DIR)
        shutil.make_archive(str(stage / "DCR-Reform-tosu"), "zip", root_dir=stage, base_dir=RELEASE_DIR)
        (stage / "LICENSE").unlink()
        stage.rename(out)
        print(out / "DCR-Reform-tosu.zip")


if __name__ == "__main__":
    main()
