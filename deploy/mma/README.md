# DCR Reform for tosu

Copy the `DCR-Reform-tosu` folder from `DCR-Reform-tosu.zip` into tosu's configured counters directory (`static/` by default). If you installed an earlier bundle, rename its `DCR Reform for tosu` folder to `DCR-Reform-tosu` and update any saved overlay URLs to `/DCR-Reform-tosu/`. The space-free folder name lets tosu route WebSocket settings to the counter.

## 切换难度算法 / Switch estimator

1. 打开 [tosu Dashboard](http://127.0.0.1:24050/)，找到 **DCR Reform**，点击这张卡片上的 **Settings**。 Open **Settings** on the **DCR Reform** card in the tosu dashboard.
2. 向下找到 **Estimator Algorithm**，选 **Sunny**、**DCR** 或 **CM3P_ext**。 Scroll to **Estimator Algorithm** and select a method.
3. 点击设置窗口底部的 **Update settings**，在确认窗口里点击 **确定**。左上角的标签会显示 **Sunny**、**DCR** 或 **CM3P**。 Click **Update settings** at the bottom of the dialog and confirm. The top-left badge shows the active method.

若想恢复右侧五格，把同一窗口的 **Top-right Content** 设为 **Difficulty**。五格从下往上对应 `low`、`low/mid`、`mid`、`mid/high`、`high`。 Set **Top-right Content** to **Difficulty** to show the five tier bars. For example, a calibrated value of `7.55` displays `3.55` on the left and `Reform 3 mid` with three lit bars on the right.

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

To regenerate the release calibration, run `python scripts/fit_reform_isotonic.py`. It fits through Delta first, with more spline intervals from 8th Dan to Delta, then extends the curve to the upper anchors. Above Epsilon, anchors have half weight and the curvature penalty is reduced to one quarter after the median Epsilon score for each method. Use `--fit isotonic` for a comparison on the revised scale or pass `--calibration` to the builder to package a separate calibration file.

The base index uses the DCR checksum whitelist. Unknown IDs, changed chart bytes, and rows without a verified checksum can be calculated on demand. The release also includes real, individually identified Reform anchor charts. Temporary IDs assigned to split anchors remain in the calibration inputs.
