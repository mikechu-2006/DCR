# DCR Reform for tosu

Copy the `DCR Reform for tosu` folder from `DCR-Reform-tosu.zip` into tosu's configured counters directory. Open the counter in the tosu dashboard and choose **Sunny**, **DCR**, or **CM3P_ext** under **Estimator Algorithm**. The card shows the selected method's Reform Dan estimate and tier.

The ZIP contains both sets of chart vectors, the DCR a20d and CM3P_ext RM-L heads, and the three Reform calibration curves. Sunny and covered 1.0× charts work with the ZIP alone. The existing host provides chart files to the overlay.

## Local CPU calculation

Run the companion from the DCR checkout before using **计算难度** on a chart the bundle does not cover. The companion binds to IPv4 loopback only. Chart bytes stay on the machine. It computes only the selected method and the browser saves the returned features in its local IndexedDB. Use the same browser profile and origin to reuse saved results after restarting.

```sh
python deploy/mma/cpu_service.py --threads 4
```

The command expects the a20d product checkpoint at `data2vec_main/runs/product/a20_full/checkpoints/last.pt`, the CM3P tokenizer and model at `.cache/cm3p_model_26cc694/`, the CM3P environment at `.cache/cm3p_venv/bin/python`, and MinaCalc 505 at `external/minacalc-standalone/build/libminacalc.so`. The Python environment for a20d needs the project's CPU research dependencies and Node.js on PATH. Use the pinned dependencies in `requirements-cm3p.txt` for CM3P, including `hydra-core`. The service verifies the encoder weights against the release manifest. Run with `--help` for path overrides or `--origin` if tosu uses another port.

首版支持 osu!mania 4K。学习型两种方法以原始谱面 MD5 和实际 rate 区分记录，接受 0.5×–2.0× 内的受支持速率，含 DT、HT 和自定义速率。改变谱面列、键数或判定的 mod 暂不进入此计算口径。CPU 编码时间随谱长和机器而变。若计算失败，卡片可重试，本机终端打印编码失败原因。

## Build

```sh
python deploy/mma/build.py
```

The builder uses the pinned MMA tree, applies `mma.patch`, then adds the browser modules and release assets. It writes the ZIP and the unpacked counter under `data/reform/mma_release/`. Use `--out` to build into a new directory. The companion setup is separate from the static counter.

The base index uses the DCR checksum whitelist. Unknown IDs, changed chart bytes, and rows without a verified checksum can be calculated on demand. The release also includes real, individually identified Reform anchor charts. Temporary IDs assigned to split anchors remain in the calibration inputs.
